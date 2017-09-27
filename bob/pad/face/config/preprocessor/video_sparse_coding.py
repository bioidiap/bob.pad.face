#!/usr/bin/env python

from bob.pad.face.preprocessor import VideoSparseCoding


#=======================================================================================
# Define instances here:

BLOCK_SIZE = 5
BLOCK_LENGTH = 10
MIN_FACE_SIZE = 50
NORM_FACE_SIZE = 64
DICTIONARY_FILE_NAMES = ["/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_front.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_hor.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_vert.hdf5"]
FRAME_STEP = 10

preprocessor = VideoSparseCoding(gblock_size = BLOCK_SIZE,
                                 block_length = BLOCK_LENGTH,
                                 min_face_size = MIN_FACE_SIZE,
                                 norm_face_size = NORM_FACE_SIZE,
                                 dictionary_file_names = DICTIONARY_FILE_NAMES,
                                 frame_step = FRAME_STEP)