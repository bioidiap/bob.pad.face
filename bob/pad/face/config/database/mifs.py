

from bob.pad.face.database.mifs import MIFSPadDatabase


# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
original_directory = "[YOUR_MIFS_DATABASE_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

original_extension = ".jpg" # extension of the data files


database = MIFSPadDatabase(
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
