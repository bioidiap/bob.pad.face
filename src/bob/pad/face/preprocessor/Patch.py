from collections import OrderedDict

from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.base.annotator.FailSafe import translate_kwargs
from bob.bio.video import VideoLikeContainer

from ..utils import extract_patches


class ImagePatches(TransformerMixin, BaseEstimator):
    """Extracts patches of images and returns it in a VideoLikeContainer. You need
    to wrap the further blocks (extractor and algorithm) that come after this
    in bob.bio.video wrappers.
    """

    def __init__(
        self, block_size, block_overlap=(0, 0), n_random_patches=None, **kwargs
    ):
        super(ImagePatches, self).__init__(**kwargs)
        self.block_size = block_size
        self.block_overlap = block_overlap
        self.n_random_patches = n_random_patches

    def transform(self, images):
        return [self.transform_one_image(img) for img in images]

    def transform_one_image(self, image):

        patches = extract_patches(
            image, self.block_size, self.block_overlap, self.n_random_patches
        )
        vc = VideoLikeContainer(patches, range(len(patches)))

        return vc


class VideoPatches(TransformerMixin, BaseEstimator):
    """Extracts patches of images from video containers and returns it in a
    VideoLikeContainer.
    """

    def __init__(
        self,
        face_cropper,
        block_size,
        block_overlap=(0, 0),
        n_random_patches=None,
        normalizer=None,
        **kwargs,
    ):
        super(VideoPatches, self).__init__(**kwargs)
        self.face_cropper = face_cropper
        self.block_size = block_size
        self.block_overlap = block_overlap
        self.n_random_patches = n_random_patches
        self.normalizer = normalizer

    def transform(self, videos, annotations=None):
        kwargs = translate_kwargs(dict(annotations=annotations), len(videos))
        return [
            self.transform_one_video(vid, **kw)
            for vid, kw in zip(videos, kwargs)
        ]

    def transform_one_video(self, frames, annotations=None):
        annotations = annotations or {}
        if self.normalizer is not None:
            annotations = OrderedDict(self.normalizer(annotations))

        all_patches = []
        for frame, index in zip(frames, frames.indices):

            # if annotations are given, and if particular frame annotations are
            # not missing we take them:
            annots = annotations.get(str(index))

            # preprocess image (by default: crops a face)
            preprocessed = self.face_cropper(frame, annots)
            if preprocessed is None:
                continue

            # extract patches
            patches = extract_patches(
                preprocessed,
                self.block_size,
                self.block_overlap,
                self.n_random_patches,
            )
            all_patches.extend(patches)

        vc = VideoLikeContainer(all_patches, range(len(all_patches)))

        if not len(vc):
            return None

        return vc
