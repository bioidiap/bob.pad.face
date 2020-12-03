from sklearn.base import TransformerMixin, BaseEstimator
import bob.pipelines as mario
from bob.pipelines.wrappers import _frmt
import logging

logger = logging.getLogger(__name__)


class VideoToFrames(TransformerMixin, BaseEstimator):
    """Expands video samples to frame-based samples only when transform is called.
    """

    def transform(self, video_samples):
        logger.debug(f"{_frmt(self)}.transform")
        output = []
        for sample in video_samples:
            annotations = getattr(sample, "annotations", {}) or {}

            # video is an instance of VideoAsArray or VideoLikeContainer
            video = sample.data
            for frame, frame_id in zip(video, video.indices):
                new_sample = mario.Sample(
                    frame,
                    frame_id=frame_id,
                    annotations=annotations.get(str(frame_id)),
                    parent=sample,
                )
                output.append(new_sample)

        return output

    def fit(self, X, y=None, **fit_params):
        return self

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
