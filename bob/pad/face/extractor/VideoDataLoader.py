#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:55:02 2017

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

import os

import bob.bio.video

#==============================================================================
# Main body of the class

class VideoDataLoader(object):
    """
    This class is designed to load video data given name of the file.
    The class is called by corresponding extractors in the experiments using
    emty(!) preprocessor. In this scenario the video data is loaded directly
    from the database, avoiding duplicate storage of non-processed data in the
    experimental directory.

    NOTE:
    To use this class in PAD experiments the command line argument
    ``--preprocessed-directory`` must point to the original database directory.
    For example:
    --preprocessed-directory <DIRECTORY_CONTAINING_REPLAY_ATTACK_DATABASE>

    At this point the class is just a collection of methods.
    """

    #==========================================================================
    def get_complete_filename(self, filename):
        """
        Get a complete filename given a filename without an extension.

        **Parameters:**

        ``filename`` : :py:class:`str`
            A name of the file containing the path, but no extension.

        **Returns:**

        ``filename_complete`` : :py:class:`str`
            A complete filename, incliding extension.
        """

        path, filename_no_ext = os.path.split(filename)

        filenames = []
        extensions = []

        for f in os.listdir(path):

            filenames.append(os.path.splitext(f)[0])
            extensions.append(os.path.splitext(f)[1])


        idx = filenames.index(filename_no_ext) # index of the file

        file_extension = extensions[idx] # get extension of the file

        filename_complete = os.path.join(path, filename_no_ext + file_extension)

        return filename_complete


    #==========================================================================
    def load_video_data(self, filename_complete):
        """
        Load video data given a complete filename.

        **Parameters:**

        ``filename_complete`` : :py:class:`str`
            A complete filename, incliding extension.

        **Returns:**

        ``video_data`` : FrameContainer
            A FrameContainer containing the loaded video data.
        """

        frame_selector = bob.bio.video.FrameSelector(selection_style = 'all') # select all frames from the video file

        video_data = frame_selector(filename_complete) # video data

        return video_data


    #==========================================================================
    def __call__(self, filename):
        """
        Load video data given a filename without an extension.

        **Parameters:**

        ``filename`` : :py:class:`str`
            A name of the file containing the path, but no extension.

        **Returns:**

        ``video_data`` : FrameContainer
            A FrameContainer containing the loaded video data.
        """

        filename_complete = self.get_complete_filename(filename)

        video_data = self.load_video_data(filename_complete)

        return video_data


