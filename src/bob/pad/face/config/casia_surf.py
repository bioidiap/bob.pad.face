"""The `CASIA-SURF`_ database for face anti-spoofing

After downloading, you can tell the bob library where the files are located
using::

    $ bob config set bob.db.casia_surf.directory /path/to/database/CASIA-SURF/
"""
from bob.pad.face.database import CasiaSurfPadDatabase

database = CasiaSurfPadDatabase()
