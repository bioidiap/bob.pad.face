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

from bob.ip.color import rgb_to_gray

from ..extractor import LBPHistogram

from ..preprocessor import ImageFaceCrop

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

    preprocessor = ImageFaceCrop(face_size = 64, rgb_output_flag = False)
    face = preprocessor(image, annotations)

    assert face.shape == (64, 64)
    assert np.sum(face) == 429158

    preprocessor = ImageFaceCrop(face_size = 64, rgb_output_flag = True)
    face = preprocessor(image, annotations)

    assert face.shape == (3, 64, 64)
    assert np.sum(face) == 1215525