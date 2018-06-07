#!/usr/bin/env python
"""
BATL Govt DB is a database for face PAD experiments.
"""

from bob.pad.face.database import BatlAggregatedPadDatabase

# Directory where the data files are stored.
# This directory is given in the .bob_bio_databases.txt file located in your home directory
ORIGINAL_DIRECTORY = "[YOUR_BATL_AGGREGATED_DB_DIRECTORY]" 
"""Value of ``~/.bob_bio_databases.txt`` for this database"""

ORIGINAL_EXTENSION = ".h5"  # extension of the data files

ANNOTATIONS_TEMP_DIR = '/idiap/temp/ageorge/BATL_aggregated/annotations_idiap_batl/ /idiap/temp/ageorge/BATL_aggregated/annotations_govtest_backup/'


_training_protocol='trainon_both_teston_gov_realgov'

_stream_type='thermal'

PROTOCOL = "nowig-{}-50-{}__baseline-{}-50-{}".format(_stream_type,_training_protocol,_stream_type,_training_protocol)

database = BatlAggregatedPadDatabase(
    protocol=PROTOCOL,
    original_directory=ORIGINAL_DIRECTORY,
    original_extension=ORIGINAL_EXTENSION,
    annotations_temp_dir=ANNOTATIONS_TEMP_DIR,
    landmark_detect_method="mtcnn",
    exlude_attacks_list = ["makeup"],
    training_depends_on_protocol=True,
)
"""The :py:class:`bob.pad.base.database.BatlGovtPadDatabase` derivative with BATL Db
database settings.

.. warning::

   This class only provides a programmatic interface to load data in an orderly
   manner, respecting usage protocols. It does **not** contain the raw
   data files. You should procure those yourself.

Notice that ``original_directory`` is set to ``[BatlGovtPadDatabase]``.
You must make sure to create ``${HOME}/.bob_bio_databases.txt`` file setting this
value to the places where you actually installed the BATL Govt database.
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

