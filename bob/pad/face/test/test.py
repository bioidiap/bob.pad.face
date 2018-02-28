#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
"""Test Units
"""
#==============================================================================
# Import what is needed here:
import numpy as np

from bob.io.base.test_utils import datafile

from bob.io.base import load

import bob.io.image  # for image loading functionality

import bob.bio.video

from bob.ip.color import rgb_to_gray

from ..extractor import LBPHistogram

from ..preprocessor import ImageFaceCrop

from ..preprocessor import VideoFaceCrop

from ..preprocessor import FrameDifference

from ..extractor import FrameDiffFeatures

from ..extractor import VideoLBPHistogram

from ..extractor import ImageQualityMeasure

from ..utils import face_detection_utils

import random


def test_detect_face_landmarks_in_image_mtcnn():

    img = load(datafile('testimage.jpg', 'bob.bio.face.test'))
    assert len(img) == 3
    annotations = face_detection_utils.detect_face_landmarks_in_image(
        img, method='mtcnn')
    assert len(annotations['landmarks']) == 68
    assert len(annotations['left_eye']) == 2
    assert len(annotations['right_eye']) == 2
    assert len(annotations['topleft']) == 2
    assert len(annotations['bottomright']) == 2

    #assert len(annotations['left_eye']) == (176, 220)


def test_detect_face_landmarks_in_video_mtcnn():

    img = load(datafile('testimage.jpg', 'bob.bio.face.test'))
    assert len(img) == 3
    frame_container = bob.bio.video.FrameContainer()
    frame_container.add(1, img)
    frame_container.add(2, img)

    annotations = face_detection_utils.detect_face_landmarks_in_video(
        frame_container, method='mtcnn')
    assert len(annotations) == 2
    assert len(annotations['1']['landmarks']) == 68


def test_detect_face_landmarks_in_image_dlib():

    img = load(datafile('testimage.jpg', 'bob.bio.face.test'))
    assert len(img) == 3
    annotations = face_detection_utils.detect_face_landmarks_in_image(
        img, method='dlib')
    assert len(annotations['landmarks']) == 68
    assert len(annotations['left_eye']) == 2
    assert len(annotations['right_eye']) == 2
    assert len(annotations['topleft']) == 2
    assert len(annotations['bottomright']) == 2

    #assert len(annotations['left_eye']) == (176, 220)


def test_detect_face_landmarks_in_video_dlib():

    img = load(datafile('testimage.jpg', 'bob.bio.face.test'))
    assert len(img) == 3
    frame_container = bob.bio.video.FrameContainer()
    frame_container.add(1, img)
    frame_container.add(2, img)

    annotations = face_detection_utils.detect_face_landmarks_in_video(
        frame_container, method='dlib')
    assert len(annotations) == 2
    assert len(annotations['1']['landmarks']) == 68


#==============================================================================
def test_lbp_histogram():
    lbp = LBPHistogram()
    img = load(datafile('testimage.jpg', 'bob.bio.face.test'))
    img = rgb_to_gray(img)
    features = lbp(img)
    reference = load(datafile('lbp.hdf5', 'bob.pad.face.test'))
    assert np.allclose(features, reference)


#==============================================================================
def test_image_face_crop():
    """
    Test ImageFaceCrop preprocessor, which is designed to crop faces in the images.
    """

    image = load(datafile('test_image.png', 'bob.pad.face.test'))
    annotations = {'topleft': (95, 155), 'bottomright': (215, 265)}

    preprocessor = ImageFaceCrop(face_size=64, rgb_output_flag=False)
    face = preprocessor(image, annotations)

    assert face.shape == (64, 64)
    assert np.sum(face) == 429158

    preprocessor = ImageFaceCrop(face_size=64, rgb_output_flag=True)
    face = preprocessor(image, annotations)

    assert face.shape == (3, 64, 64)
    assert np.sum(face) == 1215525


#==============================================================================
def convert_image_to_video_data(image, annotations, n_frames):
    """
    Convert input image to video and image annotations to frame annotations.

    **Parameters:**

    ``image`` : 2D or 3D :py:class:`numpy.ndarray`
        Input image (RGB or gray-scale).

    ``annotations`` : :py:class:`dict`
        A dictionary containing annotations of the face bounding box.
        Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col)}``

    ``n_frames`` : :py:class:`int`
        Number of frames in the output video

    **Returns:**

    ``frame_container`` : FrameContainer
        Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
        for further details.

    ``video_annotations`` : :py:class:`dict`
        A dictionary containing the annotations for each frame in the video.
        Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
        Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
        is the dictionary defining the coordinates of the face bounding box in frame N.
    """

    frame_container = bob.bio.video.FrameContainer(
    )  # initialize the FrameContainer

    video_annotations = {}

    for idx, fn in enumerate(range(0, n_frames)):

        frame_container.add(idx, image)  # add current frame to FrameContainer

        video_annotations[str(idx)] = annotations

    return frame_container, video_annotations


#==============================================================================
def test_video_face_crop():
    """
    Test VideoFaceCrop preprocessor, which is designed to crop faces in the video.
    """

    image = load(datafile('test_image.png', 'bob.pad.face.test'))
    annotations = {'topleft': (95, 155), 'bottomright': (215, 265)}

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
        mask_seed=MASK_SEED,
        check_face_size_flag=CHECK_FACE_SIZE_FLAG,
        min_face_size=MIN_FACE_SIZE,
        use_local_cropper_flag=USE_LOCAL_CROPPER_FLAG,
        color_channel=COLOR_CHANNEL)

    video, annotations = convert_image_to_video_data(image, annotations, 20)

    faces = preprocessor(frames=video, annotations=annotations)

    assert len(faces) == 20
    assert faces[0][1].shape == (64, 64)
    assert faces[-1][1].shape == (64, 64)
    assert np.sum(faces[0][1]) == 429158
    assert np.sum(faces[-1][1]) == 429158

    #==========================================================================
    # test another configuration of the  VideoFaceCrop preprocessor:

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
        rgb_output_flag=RGB_OUTPUT_FLAG,
        detect_faces_flag=DETECT_FACES_FLAG,
        face_detection_method=FACE_DETECTION_METHOD)

    video, _ = convert_image_to_video_data(image, annotations, 3)

    faces = preprocessor(frames=video, annotations=annotations)

    assert len(faces) == 3
    assert faces[0][1].shape == (3, 64, 64)
    assert faces[-1][1].shape == (3, 64, 64)
    assert np.sum(faces[0][1]) == 1238664
    assert np.sum(faces[-1][1]) == 1238664


#==============================================================================
def test_frame_difference():
    """
    Test FrameDifference preprocessor computing frame differences for both
    facial and non-facial/background regions.
    """

    image = load(datafile('test_image.png', 'bob.pad.face.test'))
    annotations = {'topleft': (95, 155), 'bottomright': (215, 265)}

    n_frames = 20

    video, annotations = convert_image_to_video_data(image, annotations,
                                                     n_frames)

    NUMBER_OF_FRAMES = None  # process all frames
    CHECK_FACE_SIZE_FLAG = True  # Check size of the face
    MIN_FACE_SIZE = 50  # Minimal size of the face to consider

    preprocessor = FrameDifference(
        number_of_frames=NUMBER_OF_FRAMES,
        check_face_size_flag=CHECK_FACE_SIZE_FLAG,
        min_face_size=MIN_FACE_SIZE)

    diff = preprocessor(frames=video, annotations=annotations)

    assert diff.shape == (n_frames - 1, 2)
    assert (diff == 0).all()


#==============================================================================
def test_frame_diff_features():
    """
    Test FrameDiffFeatures extractor computing 10 features given frame differences.
    """

    WINDOW_SIZE = 20
    OVERLAP = 0

    extractor = FrameDiffFeatures(window_size=WINDOW_SIZE, overlap=OVERLAP)

    data = np.transpose(np.vstack([range(0, 100), range(0, 100)]))

    features = extractor(data)

    assert len(features) == 5
    assert len(features[0][1]) == 10
    assert len(features[-1][1]) == 10
    assert (features[0][1][0:5] == features[0][1][5:]).all()
    assert (np.sum(features[0][1]) - 73.015116873109207) < 0.000001


#==============================================================================
def test_video_lbp_histogram():
    """
    Test VideoLBPHistogram extractor.
    """

    image = load(datafile('test_image.png', 'bob.pad.face.test'))
    annotations = {'topleft': (95, 155), 'bottomright': (215, 265)}

    CROPPED_IMAGE_SIZE = (64, 64)  # The size of the resulting face
    CROPPED_POSITIONS = {'topleft': (0, 0), 'bottomright': CROPPED_IMAGE_SIZE}
    FIXED_POSITIONS = None
    MASK_SIGMA = None  # The sigma for random values areas outside image
    MASK_NEIGHBORS = 5  # The number of neighbors to consider while extrapolating
    MASK_SEED = None  # The seed for generating random values during extrapolation
    CHECK_FACE_SIZE_FLAG = True  # Check the size of the face
    MIN_FACE_SIZE = 50  # Minimal possible size of the face
    USE_LOCAL_CROPPER_FLAG = True  # Use the local face cropping class (identical to Ivana's paper)
    RGB_OUTPUT_FLAG = False  # The output is gray-scale
    COLOR_CHANNEL = 'gray'  # Convert image to gray-scale format

    preprocessor = VideoFaceCrop(
        cropped_image_size=CROPPED_IMAGE_SIZE,
        cropped_positions=CROPPED_POSITIONS,
        fixed_positions=FIXED_POSITIONS,
        mask_sigma=MASK_SIGMA,
        mask_neighbors=MASK_NEIGHBORS,
        mask_seed=MASK_SEED,
        check_face_size_flag=CHECK_FACE_SIZE_FLAG,
        min_face_size=MIN_FACE_SIZE,
        use_local_cropper_flag=USE_LOCAL_CROPPER_FLAG,
        rgb_output_flag=RGB_OUTPUT_FLAG,
        color_channel=COLOR_CHANNEL)

    video, annotations = convert_image_to_video_data(image, annotations, 20)

    faces = preprocessor(frames=video, annotations=annotations)

    LBPTYPE = 'uniform'
    ELBPTYPE = 'regular'
    RAD = 1
    NEIGHBORS = 8
    CIRC = False
    DTYPE = None

    extractor = VideoLBPHistogram(
        lbptype=LBPTYPE,
        elbptype=ELBPTYPE,
        rad=RAD,
        neighbors=NEIGHBORS,
        circ=CIRC,
        dtype=DTYPE)

    lbp_histograms = extractor(faces)

    assert len(lbp_histograms) == 20
    assert len(lbp_histograms[0][1]) == 59
    assert (lbp_histograms[0][1] == lbp_histograms[-1][1]).all()
    assert (lbp_histograms[0][1][0] - 0.12695109261186263) < 0.000001
    assert (lbp_histograms[0][1][-1] - 0.031737773152965658) < 0.000001


#==============================================================================
def test_video_quality_measure():
    """
    Test ImageQualityMeasure with Wrapper extractor.
    """

    image = load(datafile('test_image.png', 'bob.pad.face.test'))
    annotations = {'topleft': (95, 155), 'bottomright': (215, 265)}

    video, annotations = convert_image_to_video_data(image, annotations, 2)

    GALBALLY = True
    MSU = True
    DTYPE = None

    extractor = bob.bio.video.extractor.Wrapper(ImageQualityMeasure(galbally=GALBALLY, msu=MSU, dtype=DTYPE))

    features = extractor(video)

    assert len(features) == 2
    assert len(features[0][1]) == 139
    assert (features[0][1] == features[-1][1]).all()
    assert (features[0][1][0] - 2.7748559659812599e-05) < 0.000001
    assert (features[0][1][-1] - 0.16410418866596271) < 0.000001


#==============================================================================
def convert_array_to_list_of_frame_cont(data):
    """
    Convert an input 2D array to a list of FrameContainers.

    **Parameters:**

    ``data`` : 2D :py:class:`numpy.ndarray`
        Input data array of the dimensionality (N_samples X N_features ).

        **Returns:**

    ``frame_container_list`` : [FrameContainer]
        A list of FrameContainers, see ``bob.bio.video.utils.FrameContainer``
        for further details. Each frame container contains one feature vector.
    """

    frame_container_list = []

    for idx, vec in enumerate(data):

        frame_container = bob.bio.video.FrameContainer(
        )  # initialize the FrameContainer

        frame_container.add(0, vec)

        frame_container_list.append(
            frame_container)  # add current frame to FrameContainer

    return frame_container_list
