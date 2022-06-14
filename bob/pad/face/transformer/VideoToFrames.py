import logging

from functools import partial

from sklearn.base import BaseEstimator, TransformerMixin

import bob.pipelines as mario

from bob.pipelines.wrappers import _frmt

logger = logging.getLogger(__name__)


def _get(sth):
    return sth


class VideoToFrames(TransformerMixin, BaseEstimator):
    """Expands video samples to frame-based samples only when transform is called."""

    def transform(self, video_samples):
        logger.debug(f"{_frmt(self)}.transform")
        output = []
        for sample in video_samples:
            annotations = getattr(sample, "annotations", {}) or {}

            # video is an instance of VideoAsArray or VideoLikeContainer
            video = sample.data

            for frame, frame_id in zip(video, video.indices):
                if frame is None:
                    continue
                # create a load method so that we can create DelayedSamples because
                # the input samples could be DelayedSamples with delayed attributes
                # as well and we don't want to load those delayed attributes.
                new_sample = mario.DelayedSample(
                    partial(_get, frame),
                    frame_id=frame_id,
                    annotations=annotations.get(str(frame_id)),
                    parent=sample,
                )
                output.append(new_sample)

        return output

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {"requires_fit": False, "bob_checkpoint_features": False}
