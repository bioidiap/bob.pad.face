"""Config file for the CASIA FASD dataset.
Please run ``bob config set bob.db.casia_fasd.directory /path/to/database/casia_fasd/``
in terminal to point to the original files of the dataset on your computer."""

from bob.pad.face.database import CasiaFasdPadDatabase

database = CasiaFasdPadDatabase()
