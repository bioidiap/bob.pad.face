from .FaceCropAlign import FaceCropAlign
from .FrameDifference import FrameDifference
from .VideoSparseCoding import VideoSparseCoding


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
    FaceCropAlign,
    FrameDifference,
    VideoSparseCoding,
)
__all__ = [_ for _ in dir() if not _.startswith('_')]
