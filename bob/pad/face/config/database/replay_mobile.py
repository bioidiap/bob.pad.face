#!/usr/bin/env python

"""`Replay-Mobile`_ is a database for face PAD experiments.

The Replay-Mobile Database for face spoofing consists of 1030 video clips of photo and video attack attempts to 40 clients,
under different lighting conditions.
These videos were recorded with current devices from the market -- an iPad Mini2 (running iOS) and a LG-G4 smartphone (running Android).
This Database was produced at the Idiap Research Institute (Switzerland) within the framework
of collaboration with Galician Research and Development Center in Advanced Telecommunications - Gradiant (Spain).
The reference citation is [CBVM16]_.

You can download the raw data of the `Replay-Mobile`_ database by following
the link.

.. include:: links.rst
"""

from bob.pad.face.database import ReplayMobilePadDatabase


# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
original_directory = "[YOUR_REPLAY_MOBILE_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

original_extension = ".mov" # extension of the data files


database = ReplayMobilePadDatabase(
    protocol='grandtest',
    original_directory=original_directory,
    original_extension=original_extension,
    training_depends_on_protocol=True,
)
"""The :py:class:`bob.pad.base.database.PadDatabase` derivative with Replay-Mobile
database settings.

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   data files. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_REPLAY_MOBILE_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the Replay-Mobile Database, as
explained in the section :ref:`bob.pad.face.baselines`.
"""