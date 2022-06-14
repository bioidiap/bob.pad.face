"""`Replay-Attack`_ is a database for face PAD experiments.

The Replay-Attack Database for face spoofing consists of 1300 video clips of
photo and video attack attempts to 50 clients, under different lighting
conditions. This Database was produced at the Idiap Research Institute, in
Switzerland. The reference citation is [CAM12]_.

You can download the raw data of the `Replay-Attack`_ database by following the
link. After downloading, you can tell the bob library where the files are
located using::

    $ bob config set bob.db.replay_attack.directory /path/to/database/replay/protocols/replayattack-database/
"""
from bob.pad.face.database import ReplayAttackPadDatabase

database = ReplayAttackPadDatabase()
