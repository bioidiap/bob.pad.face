"""`Replayattack`_ is a database for face PAD experiments.

The Replay-Attack Database for face spoofing consists of 1300 video clips of photo and video attack attempts to 50 clients,
under different lighting conditions. This Database was produced at the Idiap Research Institute, in Switzerland.
The reference citation is [CAM12]_.

You can download the raw data of the `Replayattack`_ database by following
the link.

.. include:: links.rst
"""
from bob.pad.face.database import ReplayPadDatabase
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector
from bob.extension import rc

database = DatabaseConnector(
    ReplayPadDatabase(
        protocol="grandtest",
        original_directory=rc.get("bob.db.replay.directory"),
        original_extension=".mov",
    )
)
