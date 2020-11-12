from .database import VideoPadFile
from .replay import ReplayPadDatabase
from .replay_mobile import ReplayMobilePadDatabase
from .mifs import MIFSPadDatabase
from .celeb_a import CELEBAPadDatabase
from .maskattack import MaskAttackPadDatabase
from .casiasurf import CasiaSurfPadDatabase
from .casiafasd import CasiaFasdPadDatabase
from .brsu import BRSUPadDatabase


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
    VideoPadFile,
    ReplayPadDatabase,
    ReplayMobilePadDatabase,
    MIFSPadDatabase,
    CELEBAPadDatabase,
    MaskAttackPadDatabase,
    CasiaSurfPadDatabase,
    CasiaFasdPadDatabase,
    BRSUPadDatabase
)

__all__ = [_ for _ in dir() if not _.startswith('_')]
