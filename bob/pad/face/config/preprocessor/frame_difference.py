#!/usr/bin/env python

from bob.pad.face.preprocessor import FrameDifference


#=======================================================================================
# Define instances here:

NUMBER_OF_FRAMES = 200 # process at most 200 frames
CHECK_FACE_SIZE_FLAG = True # Check size of the face
MIN_FACE_SIZE = 50

frame_diff_min_size_50_200_frames = FrameDifference(number_of_frames = NUMBER_OF_FRAMES,
                                                    check_face_size_flag = CHECK_FACE_SIZE_FLAG,
                                                    min_face_size = MIN_FACE_SIZE)

