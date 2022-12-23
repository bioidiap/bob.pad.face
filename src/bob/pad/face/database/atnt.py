#!/usr/bin/env python
"""
ATNT database implementation
"""


from exposed.rc import UserDefaults
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDatabase, FileSampleLoader

rc = UserDefaults("~/.bobrc", "BOBRC")


class AtntPadDatabase(CSVDatabase):
    """
    The AT&T (aka ORL) database of faces
    (http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). This
    class defines a simple protocol for training, enrollment and probe by
    splitting the few images of the database in a reasonable manner. Due to the
    small size of the database, there is only a 'dev' group, and I did not
    define an 'eval' group.
    """

    name = "atnt"
    category = "base"
    dataset_protocols_name = "atnt.tar.gz"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz",
    ]
    dataset_protocols_hash = "f529acef"

    def __init__(
        self,
        protocol="idiap_protocol",
        **kwargs,
    ):

        super().__init__(
            name=self.name,
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=rc.get(
                        "bob.bio.face.atnt.directory", ""
                    ),
                    extension=rc.get("bob.bio.face.atnt.extension", ".pgm"),
                ),
            ),
            **kwargs,
        )
