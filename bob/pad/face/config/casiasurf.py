#!/usr/bin/env python
"""CASIA-SURF is a database for face PAD experiments.

"""

from bob.pad.face.database import CasiaSurfPadDatabase

# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
original_directory = "[YOUR_CASIASURF_DB_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

original_extension = ".jpg"  # extension is not used to load the data in the HLDI of this database

database = CasiaSurfPadDatabase(
    protocol='color',
    original_directory=original_directory,
    original_extension=original_extension,
)
"""The :py:class:`bob.pad.base.database.PadDatabase` derivative with CASIA-SURF 
database settings.

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   data files. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_CASIASURF_DB_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the CASIA-SURF Database, as
explained in the section :ref:`bob.pad.face.baselines`.
"""
