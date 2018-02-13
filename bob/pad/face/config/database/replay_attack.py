#!/usr/bin/env python

"""`Replayattack`_ is a database for face PAD experiments.

The Replay-Attack Database for face spoofing consists of 1300 video clips of photo and video attack attempts to 50 clients,
under different lighting conditions. This Database was produced at the Idiap Research Institute, in Switzerland.
The reference citation is [CAM12]_.

You can download the raw data of the `Replayattack`_ database by following
the link.

.. include:: links.rst
"""

from bob.pad.face.database import ReplayPadDatabase


# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
original_directory = "[YOUR_REPLAY_ATTACK_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

original_extension = ".mov" # extension of the data files


database = ReplayPadDatabase(
    protocol='grandtest',
    original_directory=original_directory,
    original_extension=original_extension,
    training_depends_on_protocol=True,
)
"""The :py:class:`bob.pad.base.database.PadDatabase` derivative with Replayattack
database settings

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   data files. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_REPLAY_ATTACK_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the Replayattack Database, as
explained in the section :ref:`bob.pad.face.baselines`.
"""