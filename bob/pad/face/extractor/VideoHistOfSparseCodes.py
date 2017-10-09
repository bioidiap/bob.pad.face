#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:33:45 2017

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.extractor import Extractor

import numpy as np

import bob.bio.video

from bob.pad.face.extractor import ImageQualityMeasure

#==============================================================================
# Main body:

class VideoHistOfSparseCodes(Extractor, object):
    """
    This class is designed to extract histograms of sparse codes.

    **Parameters:**

    ``method`` : :py:class:`str`
        A method to use in the histogram computation. Two options are available:
        "mean" and "hist". Default: "mean".
    """

    #==========================================================================
    def __init__(self,
                 method = "mean"):

        super(VideoHistOfSparseCodes, self).__init__(method = method)

        self.method = method

        # extractor to process a single image/frame:
        extractor = ImageQualityMeasure()

        # This extractor is used only to write and read the feature vectors.
        self.video_extractor = bob.bio.video.extractor.Wrapper(extractor)


    #==========================================================================
    def comp_hist_of_sparse_codes(self, frames, method):
        """
        Compute the histograms of sparse codes.
        """

        histograms = []

        for frame_data in frames:

            frame = frame_data[1]

            if method == "mean":

                frame_codes = np.mean(frame, axis=1)

            if method == "hist":

                frame_codes = np.mean(frame!=0, axis=1)

            for idx, row in enumerate(frame_codes):

                frame_codes[idx,:] = row/np.sum(row)

            hist = frame_codes.flatten()

            histograms.append(hist)

        return histograms


    #==========================================================================
    def convert_sparse_codes_to_frame_container(self, list_of_arrays):
        """
        Convert an input list of arrays into Frame Container.

        **Parameters:**

        ``list_of_arrays`` : [:py:class:`numpy.ndarray`]
            A list of arrays.

        **Returns:**

        ``frame_container`` : FrameContainer
            FrameContainer containing the feature vectors.
        """

        frame_container = bob.bio.video.FrameContainer() # initialize the FrameContainer

        for idx, item in enumerate(list_of_arrays):

            frame_container.add(idx, item) # add frame to FrameContainer

        return frame_container


    #==========================================================================
    def reduce_features_number(self, list_of_arrays):
        """
        Reduce the number of features.
        """

        return_list = []

        for item in list_of_arrays:

            return_list.append( item[1][32:] )

        return return_list


    #==========================================================================
    def select_reconstruction_vector(self, frames, sorted_flag):
        """
        Select either sorted or non-sorted reconstruction errors.
        """

        return_list = []

        if sorted_flag:

            for item in frames:

                return_list.append( item[1][1,:] )

        else:

            for item in frames:

                return_list.append( item[1][0,:] )

#        return_list = []
#
#        for item in frames:
#
#            return_list.append( np.max(item[1], axis=1) )

        return return_list


    #==========================================================================
    def __call__(self, frames):
        """
        Extract feature vectors.

        **Parameters:**

        ``frames`` : FrameContainer or string.
            Data stored in the FrameContainer,
            see ``bob.bio.video.utils.FrameContainer`` for further details.

        **Returns:**

        ``frame_container`` : FrameContainer
            Histograms of sparse codes stored in the FrameContainer.
        """

#        histograms = self.comp_hist_of_sparse_codes(frames, self.method)

#        histograms = self.reduce_features_number(frames)

        sorted_flag = False

        list_of_error_vecs = self.select_reconstruction_vector(frames, sorted_flag)

        frame_container = self.convert_sparse_codes_to_frame_container(list_of_error_vecs)

        return frame_container


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
