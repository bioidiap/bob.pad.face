from .LBPHistogram import LBPHistogram
from .VideoLBPHistogram import VideoLBPHistogram
from .ImageQualityMeasure import ImageQualityMeasure
from .VideoDataLoader import VideoDataLoader
from .VideoQualityMeasure import VideoQualityMeasure
from .FrameDiffFeatures import FrameDiffFeatures
from .BatchAutoencoder import BatchAutoencoder


def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        The objects that you want sphinx to beleive that are defined here.

    Resolves `Sphinx referencing issues <https//github.com/sphinx-
    doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    LBPHistogram,
    VideoLBPHistogram,
    ImageQualityMeasure,
    VideoQualityMeasure,
    VideoDataLoader,
    FrameDiffFeatures,
)
__all__ = [_ for _ in dir() if not _.startswith('_')]
