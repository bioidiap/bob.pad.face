from sklearn.base import TransformerMixin, BaseEstimator
import bob.pipelines as mario
import logging

logger = logging.getLogger(__name__)


class VideoToFrames(TransformerMixin, BaseEstimator):
    """Expands frame containers to frame-based samples only when transform is called.
    When fit_transform is called, it just converts frame containers to numpy arrays.
    """

    def transform(self, video_samples):
        logger.info(
            f"Calling {self.__class__.__name__}.transform from VideoToFrames"
        )
        output = []
        for sample in video_samples:
            for frame, frame_id in zip(sample.data, sample.data.indices):
                new_sample = mario.Sample(frame, frame_id=frame_id, parent=sample)
                output.append(new_sample)
        return output

    def fit(self, X, y=None, **fit_params):
        return self

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
