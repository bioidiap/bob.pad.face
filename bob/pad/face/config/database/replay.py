#!/usr/bin/env python

from bob.pad.face.database import ReplayPadDatabase


# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
original_directory = "[YOUR_REPLAY_ATTACK_DIRECTORY]"
original_extension = ".mov" # extension of the data files


database = ReplayPadDatabase(
    protocol='grandtest',
    original_directory=original_directory,
    original_extension=original_extension,
    training_depends_on_protocol=True,
)
