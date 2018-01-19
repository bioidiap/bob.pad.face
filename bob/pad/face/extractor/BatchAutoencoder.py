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

#==============================================================================
# Main body:


class BatchAutoencoder(Extractor, object):
    """
    TODO: Doc......

    **Parameters:**

    ``a`` : :py:class:`bool`
        If ``True``, galbally features will be added to the features.
        Default: ``True``.

    ``b`` : :py:class:`bool`
        If ``True``, MSU features will be added to the features.
        Default: ``True``.

    ``c`` : numpy.dtype
        The data type of the resulting feature vector.
        Default: ``None``.
    """

    #==========================================================================
    def __init__(self, a=True, b=True, c=None, **kwargs):

        super(BatchAutoencoder, self).__init__(
            a=a, b=b, c=c)

        self.a = a
        self.b = b
        self.c = c

    #==========================================================================
    def __call__(self, frames):
        """
        TODO: Extract feature vectors containing ...... for each frame
        in the input color video sequence/container. The resulting features
        will be saved to the FrameContainer too.

        **Parameters:**

        ``frames`` : FrameContainer or string.
            Video data stored in the FrameContainer,
            see ``bob.bio.video.utils.FrameContainer`` for further details.
            If string, the name of the file to load the video data from is
            defined in it. String is possible only when empty preprocessor is
            used. In this case video data is loaded directly from the database.

        **Returns:**

        ``features`` : FrameContainer
            .....
        """

        if isinstance(frames, six.string_types):  # if frames is a path(!)

            video_loader = VideoDataLoader()

            frames = video_loader(frames)  # frames is now a FrameContainer


#        import ipdb; ipdb.set_trace()

#        TODO: OLEGS - will add a conversion of the frame container to the normalized torch tensor

#        TODO: ANJITH - loading of the Autoencoder model, passing above data through the model

#        TODO: Compute features: OLEGS

        return features

    #==========================================================================
    def write_feature(self, frames, file_name):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_data() method of the Extractor class.

        **Parameters:**

        ``frames`` :
            Data returned by the __call__ method of the class.

        ``file_name`` : :py:class:`str`
            Name of the file.
        """

        self.video_extractor.write_feature(frames, file_name)

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

        frames = self.video_extractor.read_feature(file_name)

        return frames
