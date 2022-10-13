import logging

from functools import partial

from sklearn.base import BaseEstimator, TransformerMixin

from bob.pipelines.sample import DelayedSample, Sample
from bob.pipelines.wrappers import _frmt

logger = logging.getLogger(__name__)


def _get(sth):
    return sth


class VideoToFrames(TransformerMixin, BaseEstimator):
    """Expands video samples to frame-based samples only when transform is called."""

    def __init__(self, delayed_output=True):
        self.delayed_output = delayed_output

    def transform(self, video_samples):
        logger.debug(f"{_frmt(self)}.transform")
        outputs = []
        for vid_sample in video_samples:
            annotations = getattr(vid_sample, "annotations", None)
            # Define groups with `sample.key`` since we need a unique ID for
            # each video. The `groups` attribute is used to do cross-validation
            if not hasattr(vid_sample, "key"):
                raise ValueError(
                    "Video sample must have a unique `key` "
                    "attribute to be used with {}".format(
                        self.__class__.__name__
                    )
                )
            groups = vid_sample.key
            # video is an instance of VideoAsArray or VideoLikeContainer
            video = vid_sample.data
            for frame, frame_id in zip(video, video.indices):
                if frame is None:
                    continue
                # Do we have frame annotations?
                frame_annotations = None
                if annotations is not None:
                    # Global annotation are present -> query them
                    frame_annotations = annotations.get(str(frame_id))
                # Update key, otherwise get the one from parent and each frames
                # get the same one, breaking checkpoint mechanic for steps
                # later down the pipelines
                key = "{}_{}".format(vid_sample.key, frame_id)
                if self.delayed_output:
                    # create a load method so that we can create DelayedSamples
                    # because the input samples could be DelayedSamples with
                    # delayed attributes as well and we don't want to load
                    # those delayed attributes.
                    sample = DelayedSample(
                        partial(_get, frame),
                        frame_id=frame_id,
                        video_key=groups,
                        parent=vid_sample,
                        # Override parent's attributes
                        annotations=frame_annotations,
                        key=key,
                    )
                else:
                    sample = Sample(
                        frame,
                        frame_id=frame_id,
                        video_key=groups,
                        parent=vid_sample,
                        # Override parent's attributes
                        annotations=frame_annotations,
                        key=key,
                    )
                outputs.append(sample)
        return outputs

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {
            "requires_fit": False,
            "bob_checkpoint_features": False,
        }
