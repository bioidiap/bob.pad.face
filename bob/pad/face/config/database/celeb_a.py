#!/usr/bin/env python
"""`CELEBA`_ is a face makeup spoofing database adapted for face PAD experiments.


You can download the raw data of the `CELEBA`_ database by following
the link.

.. include:: links.rst

"""

from bob.pad.face.database.celeb_a import CELEBAPadDatabase

# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
original_directory = "[YOUR_CELEB_A_DATABASE_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

original_extension = ""  # extension of the data files

database = CELEBAPadDatabase(
    protocol='grandtest',
    original_directory=original_directory,
    original_extension=original_extension,
    training_depends_on_protocol=True,
)
"""The :py:class:`bob.pad.base.database.PadDatabase` derivative with CELEBA
database settings.

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   data files. You should procure those yourself.

Notice that ``original_directory`` is set to ``[YOUR_CELEBA_DATABASE_DIRECTORY]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` setting this
value to the place where you actually installed the CELEBA Database, as
explained in the section :ref:`bob.pad.face.baselines`.
"""