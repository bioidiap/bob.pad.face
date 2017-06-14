#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:48:43 2017

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

from bob.pad.face.extractor import LBPHistogram

import bob.bio.video


#==============================================================================
# Main body:

class VideoLBPHistogram(Extractor, object):
    """
    This class is designed to extract LBP histograms for each frame in the input
    video sequence/container.

    **Parameters:**

    ``lbptype`` : :py:class:`str`
        The type of the LBP operator ("regular", "uniform" or "riu2").
        Default: uniform.

    ``elbptype`` : :py:class:`str`
        The type of extended version of LBP (regular if not extended version
        is used, otherwise transitional, direction_coded or modified).
        Default: regular.

    ``rad`` : :py:class:`float`
        The radius of the circle on which the points are taken (for circular
        LBP). Default: 1

    ``neighbors`` : :py:class:`int`
        The number of points around the central point on which LBP is
        computed. Possible options: (4, 8, 16). Default: 8.

    ``circ`` : :py:class:`bool`
        Set to True if circular LBP is needed. Default: False.

    ``dtype`` : numpy.dtype
        If specified in the constructor, the resulting features will have
        that type of data. Default: None.
    """

    #==========================================================================
    def __init__(self,
                 lbptype='uniform',
                 elbptype='regular',
                 rad=1,
                 neighbors=8,
                 circ=False,
                 dtype=None):


        super(VideoLBPHistogram, self).__init__(lbptype = lbptype,
                                                elbptype = elbptype,
                                                rad = rad,
                                                neighbors = neighbors,
                                                circ = circ,
                                                dtype = dtype)

        self.lbptype = lbptype
        self.elbptype = elbptype
        self.rad = rad
        self.neighbors = neighbors
        self.circ = circ
        self.dtype = dtype

        # extractor to process a single image/frame:
        extractor = LBPHistogram(lbptype=lbptype,
                                 elbptype=elbptype,
                                 rad=rad,
                                 neighbors=neighbors,
                                 circ=circ,
                                 dtype=dtype)

        # a wrapper allowing to apply above extractor to the whole video:
        self.video_extractor = bob.bio.video.extractor.Wrapper(extractor)


    #==========================================================================
    def __call__(self, frames):
        """
        Extracts LBP histogram for each frame in the input video sequence/container.s

        **Parameters:**

        ``frames`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.

        **Returns:**

        ``lbp_histograms`` : FrameContainer
            LBP histograms for each frame stored in the FrameContainer.
        """

        lbp_histograms = self.video_extractor(frames = frames)

        return lbp_histograms


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


