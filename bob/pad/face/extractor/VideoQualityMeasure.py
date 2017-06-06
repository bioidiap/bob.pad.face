#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:39:34 2017

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

from bob.pad.face.extractor import ImageQualityMeasure

import bob.bio.video

from bob.pad.face.extractor import VideoDataLoader

import six

#==============================================================================
# Main body:

class VideoQualityMeasure(Extractor, object):
    """
    This class is designed to extract Quality Measures for each frame in the
    input color video. For further documentation and description of features,
    see "bob.ip.qualitymeasure".

    **Parameters:**

    ``galbally`` : :py:class:`bool`
        If ``True``, galbally features will be added to the features.
        Default: ``True``.

    ``msu`` : :py:class:`bool`
        If ``True``, MSU features will be added to the features.
        Default: ``True``.

    ``dtype`` : numpy.dtype
        The data type of the resulting feature vector.
        Default: ``None``.
    """

    #==========================================================================
    def __init__(self,
                 galbally=True,
                 msu=True,
                 dtype=None,
                 **kwargs):

        super(VideoQualityMeasure, self).__init__(galbally = galbally,
                                                  msu = msu,
                                                  dtype = dtype)

        self.galbally = galbally
        self.msu = msu
        self.dtype = dtype

        # extractor to process a single image/frame:
        extractor = ImageQualityMeasure(galbally = galbally,
                                        msu = msu,
                                        dtype = dtype)

        # a wrapper allowing to apply above extractor to the whole video:
        self.video_extractor = bob.bio.video.extractor.Wrapper(extractor)


    #==========================================================================
    def __call__(self, frames):
        """
        Extract feature vectors containing Quality Measures for each frame
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

        ``quality_measures`` : FrameContainer
            Quality Measures for each frame stored in the FrameContainer.
        """

        if isinstance(frames, six.string_types): # if frames is a path(!)

            video_loader = VideoDataLoader()

            frames = video_loader(frames) # frames is now a FrameContainer

#        import ipdb; ipdb.set_trace()

        quality_measures = self.video_extractor(frames = frames)

        return quality_measures


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


