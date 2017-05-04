#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Test Units
"""
import numpy as np
from bob.io.base.test_utils import datafile
from bob.io.base import load
import bob.io.image  # for image loading functionality
from bob.ip.color import rgb_to_gray
from ..extractor import LBPHistogram


def test_lbp_histogram():
    lbp = LBPHistogram()
    img = load(datafile('testimage.jpg', 'bob.bio.face.test'))
    img = rgb_to_gray(img)
    features = lbp(img)
    reference = load(datafile('lbp.hdf5', 'bob.pad.face.test'))
    assert np.allclose(features, reference)
