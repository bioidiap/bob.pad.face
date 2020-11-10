from bob.pad.face.database import MaskAttackPadDatabase

# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
original_directory = "[YOUR_MASK_ATTACK_DIRECTORY]"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

original_extension = ".avi"  # extension is not used to load the data in the HLDI of this database

database = MaskAttackPadDatabase(
    protocol='classification',
    original_directory=original_directory,
    original_extension=original_extension,
)
