# isort: skip_file
from .database import VideoPadSample
from .casia_fasd import CasiaFasdPadDatabase
from .casia_surf import CasiaSurfPadDatabase
from .mask_attack import MaskAttackPadDatabase
from .replay_attack import ReplayAttackPadDatabase
from .replay_mobile import ReplayMobilePadDatabase
from .swan import SwanPadDatabase
from .oulu_npu import OuluNpuPadDatabase
from .atnt import AtntPadDatabase


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened. Parameters:

      *args: An iterable of objects to modify

    Resolves `Sphinx referencing issues
    <https://github.com/sphinx-doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    VideoPadSample,
    ReplayAttackPadDatabase,
    ReplayMobilePadDatabase,
    MaskAttackPadDatabase,
    CasiaSurfPadDatabase,
    CasiaFasdPadDatabase,
    SwanPadDatabase,
    OuluNpuPadDatabase,
    AtntPadDatabase,
)

__all__ = [_ for _ in dir() if not _.startswith("_")]
