#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains configurations to run LBP and SVM based face PAD baseline.
The settings are tuned for the Replay-attack database.
The idea of the algorithm is introduced in the following paper: [CAM12]_.
However some settings are different from the ones introduced in the paper.
"""

#=======================================================================================
sub_directory = 'lbp_svm'
"""
Sub-directory where results will be placed.

You may change this setting using the ``--sub-directory`` command-line option
or the attribute ``sub_directory`` in a configuration file loaded **after**
this resource.
"""

#=======================================================================================
# define preprocessor:

from ..preprocessor import VideoFaceCrop

CROPPED_IMAGE_SIZE = (64, 64)  # The size of the resulting face
CROPPED_POSITIONS = {'topleft': (0, 0), 'bottomright': CROPPED_IMAGE_SIZE}
FIXED_POSITIONS = None
MASK_SIGMA = None  # The sigma for random values areas outside image
MASK_NEIGHBORS = 5  # The number of neighbors to consider while extrapolating
MASK_SEED = None  # The seed for generating random values during extrapolation
CHECK_FACE_SIZE_FLAG = True  # Check the size of the face
MIN_FACE_SIZE = 50  # Minimal possible size of the face
USE_LOCAL_CROPPER_FLAG = True  # Use the local face cropping class (identical to Ivana's paper)
COLOR_CHANNEL = 'gray'  # Convert image to gray-scale format

preprocessor = VideoFaceCrop(
    cropped_image_size=CROPPED_IMAGE_SIZE,
    cropped_positions=CROPPED_POSITIONS,
    fixed_positions=FIXED_POSITIONS,
    mask_sigma=MASK_SIGMA,
    mask_neighbors=MASK_NEIGHBORS,
    mask_seed=None,
    check_face_size_flag=CHECK_FACE_SIZE_FLAG,
    min_face_size=MIN_FACE_SIZE,
    use_local_cropper_flag=USE_LOCAL_CROPPER_FLAG,
    color_channel=COLOR_CHANNEL)
"""
In the preprocessing stage the face is cropped in each frame of the input video given facial annotations.
The size of the face is normalized to ``cropped_image_size`` dimensions. The faces with the size
below ``min_face_size`` threshold are discarded. The preprocessor is similar to the one introduced in
[CAM12]_, which is defined by ``use_local_cropper_flag = True``.
"""

#=======================================================================================
# define extractor:

from ..extractor import LBPHistogram

from bob.bio.video.extractor import Wrapper

LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None

extractor = Wrapper(LBPHistogram(
    lbptype=LBPTYPE,
    elbptype=ELBPTYPE,
    rad=RAD,
    neighbors=NEIGHBORS,
    circ=CIRC,
    dtype=DTYPE))
"""
In the feature extraction stage the LBP histograms are extracted from each frame of the preprocessed video.

The parameters are similar to the ones introduced in [CAM12]_.
"""

#=======================================================================================
# define algorithm:

from bob.pad.base.algorithm import SVM

MACHINE_TYPE = 'C_SVC'
KERNEL_TYPE = 'RBF'
N_SAMPLES = 10000
TRAINER_GRID_SEARCH_PARAMS = {
    'cost': [2**P for P in range(-3, 14, 2)],
    'gamma': [2**P for P in range(-15, 0, 2)]
}
MEAN_STD_NORM_FLAG = True  # enable mean-std normalization
FRAME_LEVEL_SCORES_FLAG = True  # one score per frame(!) in this case

algorithm = SVM(
    machine_type=MACHINE_TYPE,
    kernel_type=KERNEL_TYPE,
    n_samples=N_SAMPLES,
    trainer_grid_search_params=TRAINER_GRID_SEARCH_PARAMS,
    mean_std_norm_flag=MEAN_STD_NORM_FLAG,
    frame_level_scores_flag=FRAME_LEVEL_SCORES_FLAG)
"""
The SVM algorithm with RBF kernel is used to classify the data into *real* and *attack* classes.
One score is produced for each frame of the input video, ``frame_level_scores_flag = True``.

In contrast to [CAM12]_, the grid search of SVM parameters is used to select the
successful settings. The grid search is done on the subset of training data. The size
of this subset is defined by ``n_samples`` parameter.

The data is also mean-std normalized, ``mean_std_norm_flag = True``.
"""
