#!/usr/bin/env python
"""
BATL Docker DB is a database for face PAD experiments.
"""
from bob.pad.face.database import BatlDockerPadDatabase
from .gt_config import GT_PATH, GT_CONFIG

# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
ORIGINAL_DIRECTORY = "/tmp/sub_dir/data/"
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

ORIGINAL_EXTENSION = ".h5"  # extension of the data files

ANNOTATIONS_TEMP_DIR = "/tmp/sub_dir/train/annotations/"

PROTOCOL = 'baseline-thermal-1'

database = BatlDockerPadDatabase(
    protocol=PROTOCOL,
    original_directory=ORIGINAL_DIRECTORY,
    original_extension=ORIGINAL_EXTENSION,
    annotations_temp_dir=ANNOTATIONS_TEMP_DIR,
    landmark_detect_method="mtcnn",
    exlude_attacks_list = ["makeup"],
    training_depends_on_protocol=True,
    ground_truth_path=GT_PATH,
    ground_truth_config=GT_CONFIG,
)

"""The :py:class:`bob.pad.base.database.BatlDockerPadDatabase` derivative with BATL Db
database settings.

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   data files. You should procure those yourself.

Notice that ``original_directory`` is set to ``[BatlDockerPadDatabase]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` file setting this
value to the places where you actually installed the BATL Docker database.
"""

protocol = PROTOCOL
"""
You may modify this at runtime by specifying the option ``--protocol`` on the
command-line of ``spoof.py`` or using the keyword ``protocol`` on a
configuration file that is loaded **after** this configuration resource.
"""

groups = ["train", "dev", "eval"]
"""The default groups to use for reproducing the baselines.

You may modify this at runtime by specifying the option ``--groups`` on the
command-line of ``spoof.py`` or using the keyword ``groups`` on a
configuration file that is loaded **after** this configuration resource.
"""

