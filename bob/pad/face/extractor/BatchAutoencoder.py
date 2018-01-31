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

from torch.autograd import Variable

import pkg_resources

import os

import importlib


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

    ``model_file``: py:class:`string`
        Absolute name of the file, containing pre-trained Network
        model. If nothing specified here, the default model corresponding to
        default ``config_file`` will be loaded.
        Default: "".

    ``code_layer_features_flag`` : :py:class:`bool`
        If ``True``, features of the code layer will be computed for each
        sample in the batch. Otherwise, MSE reconstruction error will be
        computed per sample. Default: True.

    ``config_file``: py:class:`string`
        Relative name of the config file defining the network, training data,
        and training parameters. Default: "autoencoder/autoencoder_config.py".

    ``config_group``: py:class:`string`
        Group/package name containing the configuration file. Usually all
        configs should be stored in this folder/place and there is no need to
        change this argument. Default: "bob.pad.face.config.pytorch".
    """

    #==========================================================================
    def __init__(self, model_file = "",
                 code_layer_features_flag=True,
                 config_file = "autoencoder/autoencoder_config.py",
                 config_group = "bob.pad.face.config.pytorch",
                 **kwargs):

        super(BatchAutoencoder, self).__init__(
            model_file = model_file,
            code_layer_features_flag=code_layer_features_flag,
            config_file = config_file,
            config_group = config_group)

        self.code_layer_features_flag = code_layer_features_flag
        self.config_file = config_file
        self.config_group = config_group

        relative_mod_name = '.' + os.path.splitext(self.config_file)[0].replace(os.path.sep, '.')

        config_module = importlib.import_module(relative_mod_name, self.config_group)

        self.img_transform = config_module.transform

        # Initialize the model
        self.model = config_module.Network()

        if model_file == "":

            model_file = pkg_resources.resource_filename('bob.pad.face.config.pytorch', 'autoencoder/autoencoder_model.pth')

        self.model_file = model_file

        model_state = torch.load(self.model_file)

        # Initialize the state of the model:
        self.model.load_state_dict(model_state)
        # Model is used for evaluation only
        self.model.train(False)

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

        video_tnsr = self.apply_transforms(video_data_array=video_data_array,
                                           img_transform=self.img_transform)

        # Above data can now be passed through the model.
        # Model running, encoding and reconstruction.
        reconstructed = self.model.forward(Variable(video_tnsr))
        encoded = self.model.encoder(Variable(video_tnsr))

        # Getting numpy arrays for feature extraction.
        reconstructed_numpy = reconstructed.data.numpy()
        encoded_numpy = encoded.data.numpy()
        orig_numpy = video_tnsr.numpy()  # Check

        # Feature extraction

        if self.code_layer_features_flag:  # 1. Code layer features

            features = np.reshape(encoded_numpy, (encoded_numpy.shape[0], encoded_numpy.shape[1]*encoded_numpy.shape[2]*encoded_numpy.shape[3]))

        else:  # MSE as a feature

            features = -1*((orig_numpy - reconstructed_numpy) ** 2).mean(axis=1).mean(axis=1).mean(axis=1)

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
