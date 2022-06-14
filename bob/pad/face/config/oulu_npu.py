"""The `OULU-NPU`_ Database.
A mobile face presentation attack database with real-world variations database.

To configure the location of the database on your computer, run::

    bob config set bob.db.oulu_npu.directory /path/to/database/oulu-npu


If you use this database, please cite the following publication::

    @INPROCEEDINGS{OULU_NPU_2017,
             author = {Boulkenafet, Z. and Komulainen, J. and Li, Lei. and Feng, X. and Hadid, A.},
           keywords = {biometrics, face recognition, anti-spoofing, presentation attack, generalization, colour texture},
              month = May,
              title = {{OULU-NPU}: A mobile face presentation attack database with real-world variations},
            journal = {IEEE International Conference on Automatic Face and Gesture Recognition},
               year = {2017},
    }
"""
from bob.pad.face.database import OuluNpuPadDatabase

database = OuluNpuPadDatabase()
