#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#==============================================================================
# Import what is needed here:

from __future__ import division
from bob.ip.qualitymeasure import galbally_iqm_features as iqm
from bob.ip.qualitymeasure import msu_iqa_features as iqa
import logging
import numpy as np
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)

def iqm_features(images, galbally=True, msu=True, dtype=None):
    if not (galbally or msu):
        raise ValueError("At least galbally or msu needs to be True.")

    all_features = []
    for data in images:
        assert isinstance(data, np.ndarray)

        features = []

        if galbally:

            try:

                gf_set = iqm.compute_quality_features(data)
                gf_set = np.nan_to_num(gf_set)
                features = np.hstack((features, gf_set))

            except Exception:

                logger.error(
                    "Failed to extract galbally features.", exc_info=True)

                features = np.zeros((18,))

        if msu:

            try:

                msuf_set = iqa.compute_msu_iqa_features(data)
                msuf_set = np.nan_to_num(msuf_set)
                features = np.hstack((features, msuf_set))

            except Exception:

                logger.error("Failed to extract MSU features.", exc_info=True)

                features = np.zeros((121,))

        if dtype is not None:

            features = features.astype(dtype)
        all_features.append(features)

    return np.array(all_features)

def ImageQualityMeasure(galbally=True, msu=True, dtype=None, **kwargs):
    kw_args = dict(galbally=galbally, msu=msu, dtype=dtype)
    return FunctionTransformer(iqm_features, validate=False, kw_args=kw_args)

# class ImageQualityMeasure(Extractor):
#     """
#     This class is designed to extract Image Quality Measures given input RGB
#     image. For further documentation and description of features,
#     see "bob.ip.qualitymeasure".

#     **Parameters:**

#     ``galbally`` : :py:class:`bool`
#         If ``True``, galbally features will be added to the features.
#         Default: ``True``.

#     ``msu`` : :py:class:`bool`
#         If ``True``, MSU features will be added to the features.
#         Default: ``True``.

#     ``dtype`` : np.dtype
#         The data type of the resulting feature vector.
#         Default: ``None``.
#     """

#     #==========================================================================
#     def __init__(self, galbally=True, msu=True, dtype=None, **kwargs):

#         Extractor.__init__(
#             self, galbally=galbally, msu=msu, dtype=dtype, **kwargs)

#         self.dtype = dtype
#         self.galbally = galbally
#         self.msu = msu

#     #==========================================================================
#     def __call__(self, data):
#         """
#         Compute Image Quality Measures given input RGB image.

#         **Parameters:**

#         ``data`` : 3D :py:class:`np.ndarray`
#             Input RGB image of the dimensionality (3, Row, Col), as returned
#             by Bob image loading routines.

#         **Returns:**

#         ``features`` : 1D :py:class:`np.ndarray`
#             Feature vector containing Image Quality Measures.
#         """

#         assert isinstance(data, np.ndarray)
#         assert self.galbally or self.msu

#         features = []

#         if self.galbally:

#             try:

#                 gf_set = iqm.compute_quality_features(data)
#                 gf_set = np.nan_to_num(gf_set)
#                 features = np.hstack((features, gf_set))

#             except Exception as e:

#                 logger.error(
#                     "Failed to extract galbally features.", exc_info=e)

#                 return None

#         if self.msu:

#             try:

#                 msuf_set = iqa.compute_msu_iqa_features(data)
#                 msuf_set = np.nan_to_num(msuf_set)
#                 features = np.hstack((features, msuf_set))

#             except Exception as e:

#                 logger.error("Failed to extract MSU features.", exc_info=e)

#                 return None

#         if self.dtype is not None:

#             features = features.astype(self.dtype)

#         return features
