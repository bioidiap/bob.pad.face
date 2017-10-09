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


#=======================================================================================


BLOCK_SIZE = 5
BLOCK_LENGTH = 10
MIN_FACE_SIZE = 50
NORM_FACE_SIZE = 64
DICTIONARY_FILE_NAMES = ["/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_front_10_5_16.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_hor_10_5_16.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_vert_10_5_16.hdf5"]
FRAME_STEP = 2
EXTRACT_HISTOGRAMS_FLAG = True

preprocessor_10_5_16 = VideoSparseCoding(gblock_size = BLOCK_SIZE,
                                         block_length = BLOCK_LENGTH,
                                         min_face_size = MIN_FACE_SIZE,
                                         norm_face_size = NORM_FACE_SIZE,
                                         dictionary_file_names = DICTIONARY_FILE_NAMES,
                                         frame_step = FRAME_STEP,
                                         extract_histograms_flag = EXTRACT_HISTOGRAMS_FLAG)

BLOCK_SIZE = 5
BLOCK_LENGTH = 10
MIN_FACE_SIZE = 50
NORM_FACE_SIZE = 64
DICTIONARY_FILE_NAMES = ["/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_front_10_5_32.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_hor_10_5_32.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_vert_10_5_32.hdf5"]
FRAME_STEP = 2
EXTRACT_HISTOGRAMS_FLAG = True

preprocessor_10_5_32 = VideoSparseCoding(gblock_size = BLOCK_SIZE,
                                         block_length = BLOCK_LENGTH,
                                         min_face_size = MIN_FACE_SIZE,
                                         norm_face_size = NORM_FACE_SIZE,
                                         dictionary_file_names = DICTIONARY_FILE_NAMES,
                                         frame_step = FRAME_STEP,
                                         extract_histograms_flag = EXTRACT_HISTOGRAMS_FLAG)

BLOCK_SIZE = 5
BLOCK_LENGTH = 10
MIN_FACE_SIZE = 50
NORM_FACE_SIZE = 64
DICTIONARY_FILE_NAMES = ["/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_front_10_5_64.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_hor_10_5_64.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_vert_10_5_64.hdf5"]
FRAME_STEP = 2
EXTRACT_HISTOGRAMS_FLAG = True

preprocessor_10_5_64 = VideoSparseCoding(gblock_size = BLOCK_SIZE,
                                         block_length = BLOCK_LENGTH,
                                         min_face_size = MIN_FACE_SIZE,
                                         norm_face_size = NORM_FACE_SIZE,
                                         dictionary_file_names = DICTIONARY_FILE_NAMES,
                                         frame_step = FRAME_STEP,
                                         extract_histograms_flag = EXTRACT_HISTOGRAMS_FLAG)


BLOCK_SIZE = 5
BLOCK_LENGTH = 10
MIN_FACE_SIZE = 50
NORM_FACE_SIZE = 64
DICTIONARY_FILE_NAMES = ["/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_front_10_5_128.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_hor_10_5_128.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_vert_10_5_128.hdf5"]
FRAME_STEP = 2
EXTRACT_HISTOGRAMS_FLAG = True

preprocessor_10_5_128 = VideoSparseCoding(gblock_size = BLOCK_SIZE,
                                         block_length = BLOCK_LENGTH,
                                         min_face_size = MIN_FACE_SIZE,
                                         norm_face_size = NORM_FACE_SIZE,
                                         dictionary_file_names = DICTIONARY_FILE_NAMES,
                                         frame_step = FRAME_STEP,
                                         extract_histograms_flag = EXTRACT_HISTOGRAMS_FLAG)

#=======================================================================================


BLOCK_SIZE = 5
BLOCK_LENGTH = 10
MIN_FACE_SIZE = 50
NORM_FACE_SIZE = 64
DICTIONARY_FILE_NAMES = ["/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_front_10_5_64.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_hor_10_5_64.hdf5",
                         "/idiap/user/onikisins/Projects/ODIN/Python/scripts/test_scripts/data/dictionary_vert_10_5_64.hdf5"]

FRAME_STEP = 50 # (!) a small number of feature vectors will be computed
EXTRACT_HISTOGRAMS_FLAG = True
COMP_RECONSTRUCT_ERR_FLAG = True

preprocessor_10_5_64_rec_err = VideoSparseCoding(gblock_size = BLOCK_SIZE,
                                                 block_length = BLOCK_LENGTH,
                                                 min_face_size = MIN_FACE_SIZE,
                                                 norm_face_size = NORM_FACE_SIZE,
                                                 dictionary_file_names = DICTIONARY_FILE_NAMES,
                                                 frame_step = FRAME_STEP,
                                                 extract_histograms_flag = EXTRACT_HISTOGRAMS_FLAG,
                                                 comp_reconstruct_err_flag = COMP_RECONSTRUCT_ERR_FLAG)

