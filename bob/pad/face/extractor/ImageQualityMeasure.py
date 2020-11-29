import logging

import numpy as np
from bob.ip.qualitymeasure import galbally_iqm_features as iqm
from bob.ip.qualitymeasure import msu_iqa_features as iqa
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

                logger.error("Failed to extract galbally features.", exc_info=True)

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
