#!/usr/bin/env python
# encoding: utf-8
# sushil bhattacharjee <sushil.bhattacharjee@idiap.ch>
# Fri. 10 March 2017

"""Unit-tests for bob.ip.qualitymeasure"""

import os
import numpy as np
import nose.tools
import pkg_resources
import h5py

import bob.io.base
import bob.io.base.test_utils

# import bob.io.video
# import bob.ip.color
from bob.io.image import to_bob
import imageio

from bob.pad.face.qualitymeasure import galbally_iqm_features as iqm
from bob.pad.face.qualitymeasure import msu_iqa_features as iqa


REF_VIDEO_FILE = "real_client001_android_SD_scene01.mp4"
REF_FEATURE_FILE = "real_client001_android_SD_scene01_ref_feats.h5"


def F(n):
    return pkg_resources.resource_filename(__name__, os.path.join("data", n))


input_video_file = F(REF_VIDEO_FILE)
assert os.path.isfile(input_video_file), "File: not found: %s" % input_video_file
video_data = to_bob(np.array(list((imageio.get_reader(input_video_file).iter_data()))))

numframes = 3


def load_reference_features():
    ref_feat_file = F(REF_FEATURE_FILE)
    assert os.path.isfile(ref_feat_file), "File: not found: %s" % ref_feat_file

    rf = h5py.File(ref_feat_file)
    # assert rf.has_key("/bobiqm"), "Key: /bobiqm not found in file %s" % ref_feat_file
    # assert rf.has_key("/msuiqa"), "Key: /msuiqa not found in file %s" % ref_feat_file
    galbally_ref_features = np.array(rf["bobiqm"])
    msu_ref_features = np.array(rf["msuiqa"])
    del rf
    return (galbally_ref_features, msu_ref_features)


# load reference-features into global vars.
galbally_ref_features, msu_ref_features = load_reference_features()


def test_galbally_feat_extr():

    # change this, if you add more features to galbally_iqm_features module.
    iqm_len = 18
    # feature-array to hold features for several frames
    bobfset = np.zeros([numframes, iqm_len])
    f = 0

    # process first frame separately, to get the no. of iqm features
    rgbFrame = video_data[f]
    iqmSet = iqm.compute_quality_features(rgbFrame)
    numIQM = len(iqmSet)

    # test: check that numIQM is the same as expected iqm_len (18)
    nose.tools.eq_(numIQM, iqm_len)

    # store features for first frame in feature-array
    bobfset[f] = iqmSet

    # now store iqm features for remaining test-frames of input video.
    for f in range(1, numframes):
        rgbFrame = video_data[f]
        bobfset[f] = iqm.compute_quality_features(rgbFrame)

    # test: compare feature-values in bobfset[] with those loaded from hdf5 file
    nose.tools.assert_true((bobfset == galbally_ref_features).all())
    # np.allclose(A,B)


def test_msu_feat_extr():
    # change this, if you change the no. of features in msu_iqa_features module.
    iqa_len = 121
    # feature-array to hold features for several frames
    msufset = np.zeros([numframes, iqa_len])
    f = 0

    # process first frame separately, to get the no. of iqa features
    rgbFrame = video_data[f]

    iqaSet = iqa.compute_msu_iqa_features(rgbFrame)
    numIQA = len(iqaSet)

    # test: check that numIQA matches the expected iqa_len(121)
    nose.tools.eq_(numIQA, iqa_len)

    # store features for first frame in feature-array
    msufset[f] = iqaSet

    # now store iqm features for remaining test-frames of input video.
    for f in range(1, numframes):

        rgbFrame = video_data[f]
        msuQFeats = iqa.compute_msu_iqa_features(rgbFrame)
        msufset[f] = msuQFeats

    # test: compare feature-values in bobfset[] with those loaded from hdf5 file
    np.testing.assert_allclose(msufset, msu_ref_features)
