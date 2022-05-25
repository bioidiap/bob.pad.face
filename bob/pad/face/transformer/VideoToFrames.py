import logging

from sklearn.base import BaseEstimator, TransformerMixin

import bob.pipelines as mario

from bob.pipelines.wrappers import _frmt

logger = logging.getLogger(__name__)


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
                kw = (
                    {"key": f"{sample.key}_{frame_id}"}
                    if hasattr(sample, "key")
                    else {}
                )
                new_sample = mario.Sample(
                    frame,
                    frame_id=frame_id,
                    annotations=annotations.get(str(frame_id)),
                    parent=sample,
                    **kw,
                )
                output.append(new_sample)

        return output

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {"requires_fit": False, "bob_checkpoint_features": False}
