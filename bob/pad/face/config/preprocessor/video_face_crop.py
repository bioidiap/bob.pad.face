#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from bob.pad.face.preprocessor import VideoFaceCrop

#=======================================================================================
# Define instances here:

CROPPED_IMAGE_SIZE = (64, 64)  # The size of the resulting face
CROPPED_POSITIONS = {'topleft': (0, 0), 'bottomright': CROPPED_IMAGE_SIZE}
FIXED_POSITIONS = None
MASK_SIGMA = None  # The sigma for random values areas outside image
MASK_NEIGHBORS = 5  # The number of neighbors to consider while extrapolating
MASK_SEED = None  # The seed for generating random values during extrapolation
CHECK_FACE_SIZE_FLAG = True  # Check the size of the face
MIN_FACE_SIZE = 50
USE_LOCAL_CROPPER_FLAG = True  # Use the local face cropping class (identical to Ivana's paper)
RGB_OUTPUT_FLAG = True  # Return RGB cropped face using local cropper
DETECT_FACES_FLAG = True  # find annotations locally replacing the database annotations
FACE_DETECTION_METHOD = "dlib"

rgb_face_detector_dlib = VideoFaceCrop(
    cropped_image_size=CROPPED_IMAGE_SIZE,
    cropped_positions=CROPPED_POSITIONS,
    fixed_positions=FIXED_POSITIONS,
    mask_sigma=MASK_SIGMA,
    mask_neighbors=MASK_NEIGHBORS,
    mask_seed=None,
    check_face_size_flag=CHECK_FACE_SIZE_FLAG,
    min_face_size=MIN_FACE_SIZE,
    use_local_cropper_flag=USE_LOCAL_CROPPER_FLAG,
    rgb_output_flag=RGB_OUTPUT_FLAG,
    detect_faces_flag=DETECT_FACES_FLAG,
    face_detection_method=FACE_DETECTION_METHOD)

CROPPED_IMAGE_SIZE = (64, 64)  # The size of the resulting face
CROPPED_POSITIONS = {'topleft': (0, 0), 'bottomright': CROPPED_IMAGE_SIZE}
FIXED_POSITIONS = None
MASK_SIGMA = None  # The sigma for random values areas outside image
MASK_NEIGHBORS = 5  # The number of neighbors to consider while extrapolating
MASK_SEED = None  # The seed for generating random values during extrapolation
CHECK_FACE_SIZE_FLAG = True  # Check the size of the face
MIN_FACE_SIZE = 50
USE_LOCAL_CROPPER_FLAG = True  # Use the local face cropping class (identical to Ivana's paper)
RGB_OUTPUT_FLAG = True  # Return RGB cropped face using local cropper
DETECT_FACES_FLAG = True  # find annotations locally replacing the database annotations
FACE_DETECTION_METHOD = "mtcnn"

rgb_face_detector_mtcnn = VideoFaceCrop(
    cropped_image_size=CROPPED_IMAGE_SIZE,
    cropped_positions=CROPPED_POSITIONS,
    fixed_positions=FIXED_POSITIONS,
    mask_sigma=MASK_SIGMA,
    mask_neighbors=MASK_NEIGHBORS,
    mask_seed=None,
    check_face_size_flag=CHECK_FACE_SIZE_FLAG,
    min_face_size=MIN_FACE_SIZE,
    use_local_cropper_flag=USE_LOCAL_CROPPER_FLAG,
    rgb_output_flag=RGB_OUTPUT_FLAG,
    detect_faces_flag=DETECT_FACES_FLAG,
    face_detection_method=FACE_DETECTION_METHOD)
