#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
"""Test Units
"""
# ==============================================================================
# Import what is needed here:
import numpy as np

from bob.io.base.test_utils import datafile

from bob.io.base import load

import bob.io.image  # for image loading functionality

import bob.bio.video

from bob.ip.color import rgb_to_gray

from bob.pad.face.extractor import LBPHistogram, ImageQualityMeasure


def test_lbp_histogram():
    lbp = LBPHistogram()
    img = load(datafile("testimage.jpg", "bob.bio.face.test"))
    img = rgb_to_gray(img)
    features = lbp.transform([img])[0]
    reference = load(datafile("lbp.hdf5", "bob.pad.face.test"))
    assert np.allclose(features, reference)


def notest_video_lbp_histogram():
    """
    Test LBPHistogram with Wrapper extractor.
    """

    from ..preprocessor import FaceCropAlign
    from bob.bio.video.preprocessor import Wrapper

    FACE_SIZE = 64  # The size of the resulting face
    RGB_OUTPUT_FLAG = False  # Gray-scale output
    USE_FACE_ALIGNMENT = False  # use annotations
    MAX_IMAGE_SIZE = None  # no limiting here
    FACE_DETECTION_METHOD = None  # use annotations
    MIN_FACE_SIZE = 50  # skip small faces

    image_preprocessor = FaceCropAlign(
        face_size=FACE_SIZE,
        rgb_output_flag=RGB_OUTPUT_FLAG,
        use_face_alignment=USE_FACE_ALIGNMENT,
        max_image_size=MAX_IMAGE_SIZE,
        face_detection_method=FACE_DETECTION_METHOD,
        min_face_size=MIN_FACE_SIZE,
    )

    preprocessor = Wrapper(image_preprocessor)

    image = load(datafile("test_image.png", "bob.pad.face.test"))
    annotations = {"topleft": (95, 155), "bottomright": (215, 265)}

    video, annotations = convert_image_to_video_data(image, annotations, 20)

    faces = preprocessor(frames=video, annotations=annotations)

    LBPTYPE = "uniform"
    ELBPTYPE = "regular"
    RAD = 1
    NEIGHBORS = 8
    CIRC = False
    DTYPE = None

    extractor = bob.bio.video.extractor.Wrapper(
        LBPHistogram(
            lbptype=LBPTYPE,
            elbptype=ELBPTYPE,
            rad=RAD,
            neighbors=NEIGHBORS,
            circ=CIRC,
            dtype=DTYPE,
        )
    )

    lbp_histograms = extractor(faces)

    assert len(lbp_histograms) == 20
    assert len(lbp_histograms[0][1]) == 59
    assert (lbp_histograms[0][1] == lbp_histograms[-1][1]).all()
    assert (lbp_histograms[0][1][0] - 0.12695109261186263) < 0.000001
    assert (lbp_histograms[0][1][-1] - 0.031737773152965658) < 0.000001


# ==============================================================================
def notest_video_quality_measure():
    """
    Test ImageQualityMeasure with Wrapper extractor.
    """

    image = load(datafile("test_image.png", "bob.pad.face.test"))
    annotations = {"topleft": (95, 155), "bottomright": (215, 265)}

    video, annotations = convert_image_to_video_data(image, annotations, 2)

    GALBALLY = True
    MSU = True
    DTYPE = None

    extractor = bob.bio.video.extractor.Wrapper(
        ImageQualityMeasure(galbally=GALBALLY, msu=MSU, dtype=DTYPE)
    )

    features = extractor(video)

    assert len(features) == 2
    assert len(features[0][1]) == 139
    assert (features[0][1] == features[-1][1]).all()
    assert (features[0][1][0] - 2.7748559659812599e-05) < 0.000001
    assert (features[0][1][-1] - 0.16410418866596271) < 0.000001
