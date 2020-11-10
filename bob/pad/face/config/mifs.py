"""`MIFS`_ is a face makeup spoofing database adapted for face PAD experiments.

Database assembled from a dataset consisting of 107 makeup-transformations taken
from random YouTube makeup video tutorials, adapted in this package for face-PAD
experiments. The public version of the database contains 107 such transformations
with each time two images of a subject before makeup, two images of the same
subject after makeup and two images of the target identity. For this package, a
subset of 104 makeup transformations is selected, the target identities images
discarded and the remaining images randomly distributed in three sets. More
information can be found in the reference [CDSR17]_.

You can download the raw data of the `MIFS`_ database by following
the link.

.. include:: links.rst

"""
from bob.pad.face.database import MIFSPadDatabase
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector
from bob.extension import rc

database = DatabaseConnector(
    MIFSPadDatabase(
        protocol="grandtest",
        original_directory=rc.get("bob.db.mifs.directory"),
        original_extension="",
    )
)
