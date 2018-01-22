#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 08:14:40 2018

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

import bob.bio.video

from bob.pad.face.extractor import VideoDataLoader

import six

import numpy as np

import torch

#import torchvision

import PIL

from torchvision import transforms

from torch.autograd import Variable

from torch import nn

import pkg_resources


#==============================================================================
# Main body:

class BatchAutoencoder(Extractor, object):
    """
    This class is designed to pass the input batch of images through the
    autoencoder, and compute the feature vector. Feature vectors of the
    following types can be computed for each sample in the batch:
    1. features of the code layer,
    2. reconstruction error.

    **Parameters:**

    ``code_layer_features_flag`` : :py:class:`bool`
        If ``True``, features of the code layer will be computed for each
        sample in the batch. Otherwise, MSE reconstruction error will be
        computed per sample.
    """

    #==========================================================================
    def __init__(self, code_layer_features_flag=True, **kwargs):

        super(BatchAutoencoder, self).__init__(
                code_layer_features_flag = code_layer_features_flag)

        self.code_layer_features_flag = code_layer_features_flag

        self.img_transform = transforms.Compose([transforms.Resize((64, 64)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])

        # The model class is defined here:
        # TODO: move this class to different place/config file.
        class autoencoder(nn.Module):

            def __init__(self):
                super(autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, stride=3, padding=1),  # b, 16, 10, 10
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
                    nn.Conv2d(64, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(8, 64, 3, stride=2),  # b, 16, 5, 5
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
                    nn.ReLU(True),
                    nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
                    nn.Tanh()
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        # Initialize the model
        self.model = autoencoder()

        # TODO: move the model to different place:
        model_file = pkg_resources.resource_filename('bob.pad.face', 'extractor/conv_autoencoder119.pth')

        model_state=torch.load(model_file)

        # Initialize the state of the model:
        self.model.load_state_dict(model_state)


    #==========================================================================
    def convert_and_swap_color_frames_to_array(self, frames):
        """
        Convert FrameContainer containing color video to the 4D numpy array.
        Also, the dimensionality of the data is chenged to the following format:
        num_frames x width x hight x color_channels (RGB order).

        **Parameters:**

        ``frames`` : FrameContainer.
            Video data stored in the FrameContainer,
            see ``bob.bio.video.utils.FrameContainer`` for further details.

        **Returns:**

        ``video_data_array_plt`` : 4D :py:class:`numpy.ndarray`
            Output 4D array of the size:
            num_frames x width x hight x color_channels (RGB order).
        """

        video_data_array = frames.as_array()

        video_data_array_plt = np.swapaxes(video_data_array, 1, 2)
        video_data_array_plt = np.swapaxes(video_data_array_plt, 2, 3)

        return video_data_array_plt


    #==========================================================================
    def apply_transforms(self, video_data_array, img_transform):
        """
        Apply composed transformation to each frame in the input 4D array.

        **Parameters:**

        ``video_data_array`` : 4D :py:class:`numpy.ndarray`
            A 4D array containing the color video. The size is:
            num_frames x width x hight x color_channels (RGB order).

        **Returns:**

        ``video_tnsr`` : torch FloatTensor
            Tensor containing normalized color video. The shape is:
            num_frames x color_channels (RGB order) x width x hight .
        """

        tnsr_img_list = []

        for color_img_plt in video_data_array:

            pil_img = PIL.Image.fromarray(color_img_plt)

            tnsr_img_transf = img_transform(pil_img)

            tnsr_img_list.append(tnsr_img_transf)

        video_tnsr = torch.stack(tnsr_img_list)

        return video_tnsr


    #==========================================================================
    def convert_arr_to_frame_cont(self, data):
        """
        This function converts an array of samples into a FrameContainer, where
        each frame stores features of a particular sample.

        **Parameters:**

        ``data`` : 2D :py:class:`numpy.ndarray`
            An input array of features of the size
            (Nr. of samples X Nr. of features).

        **Returns:**

        ``frames`` : FrameContainer
            Resulting FrameContainer, where each frame stores features of
            a particular sample.
        """

        frames = bob.bio.video.FrameContainer(
        )  # initialize the FrameContainer

        for idx, sample in enumerate(data):

            frames.add(idx, sample)

        return frames


    #==========================================================================
    def __call__(self, frames):
        """
        Extract feature vectors containing either code layer features, or
        reconstruction error as a feature, for each frame in the input color
        video sequence/container. The resulting features will be saved to
        the FrameContainer too.

        **Parameters:**

        ``frames`` : FrameContainer or string.
            Video data stored in the FrameContainer,
            see ``bob.bio.video.utils.FrameContainer`` for further details.
            If string, the name of the file to load the video data from is
            defined in it. String is possible only when empty preprocessor is
            used. In this case video data is loaded directly from the database.

        **Returns:**

        ``features`` : FrameContainer
            If ``self.code_layer_features_flag=True``, features of the code
            layer will be returned for each frame in the batch.
            Otherwise, MSE reconstruction error will be returned per sample.
        """

        if isinstance(frames, six.string_types):  # if frames is a path(!)

            video_loader = VideoDataLoader()

            frames = video_loader(frames)  # frames is now a FrameContainer

        video_data_array = self.convert_and_swap_color_frames_to_array(frames)

        video_tnsr = self.apply_transforms(video_data_array = video_data_array,
                                           img_transform = self.img_transform)

        # Above data can now be passed through the model.
        # Model running, encoding and reconstruction.
        reconstructed = self.model.forward(Variable(video_tnsr))
        encoded = self.model.encoder(Variable(video_tnsr))

        # Getting numpy arrays for feature extraction.
        reconstructed_numpy=reconstructed.data.numpy()
        encoded_numpy=encoded.data.numpy()
        orig_numpy=video_tnsr.numpy() #Check

        # Feature extraction

        if self.code_layer_features_flag: # 1. Code layer features

            features=np.reshape(encoded_numpy,(encoded_numpy.shape[0],encoded_numpy.shape[1]*encoded_numpy.shape[2]*encoded_numpy.shape[3]))

        else: # MSE as a feature

            features = ((orig_numpy - reconstructed_numpy) ** 2).mean(axis=1).mean(axis=1).mean(axis=1)

        features = self.convert_arr_to_frame_cont(features)

        return features


    #==========================================================================
    def write_feature(self, frames, file_name):
        """
        Writes the given data (that has been generated using the __call__
        function of this class) to file.
        This method overwrites the write_data() method of the Extractor class.

        **Parameters:**

        ``frames`` :
            Data returned by the __call__ method of the class.

        ``file_name`` : :py:class:`str`
            Name of the file.
        """

        bob.bio.video.extractor.Wrapper(Extractor()).write_feature(
            frames, file_name)


    #==========================================================================
    def read_feature(self, file_name):
        """
        Reads the preprocessed data from file.
        This method overwrites the read_data() method of the Extractor class.

        **Parameters:**

        ``file_name`` : :py:class:`str`
            Name of the file.

        **Returns:**

        ``frames`` : :py:class:`bob.bio.video.FrameContainer`
            Frames stored in the frame container.
        """

        frames = bob.bio.video.extractor.Wrapper(
            Extractor()).read_feature(file_name)

        return frames

