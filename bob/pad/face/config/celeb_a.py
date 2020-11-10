"""`CELEBA`_ is a face makeup spoofing database adapted for face PAD experiments.


You can download the raw data of the `CELEBA`_ database by following
the link.

.. include:: links.rst

"""

from bob.extension import rc
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector
from bob.pad.face.database.celeb_a import CELEBAPadDatabase

database = DatabaseConnector(
    CELEBAPadDatabase(
        protocol="grandtest",
        original_directory=rc.get("bob.db.celeba.directory"),
        original_extension="",
        training_depends_on_protocol=True,
    )
)
