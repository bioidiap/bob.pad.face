#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.bio.base.database import AnnotationsLoader
from bob.pad.base.database import CSVPADDataset, CSVToSampleLoader
import bob.io.base
from bob.extension import rc
from bob.extension.download import get_file


filename = "replat-attack.tar.gz"


class ReplayAttackDatabase(CSVPADDataset):
    def __init__(self, protocol):

        # Downloading model if not exists
        urls = [
            "https://www.idiap.ch/software/bob/databases/latest/replay-attack.tar.gz",
            "http://www.idiap.ch/software/bob/databases/latest/replay-attack.tar.gz",
        ]

        final_filename = get_file(filename, urls)

        self.annotation_type = ("bounding-box",)
        self.fixed_positions = None

        super().__init__(
            final_filename,
            protocol,
            csv_to_sample_loader=CSVToSampleLoader(
                data_loader=bob.io.base.load,
                metadata_loader=AnnotationsLoader(
                    annotation_directory=rc["bob.db.replay.annotation_dir"],
                    annotation_extension=".json",
                    annotation_type="json",
                ),
                dataset_original_directory=rc["bob.db.replay.directory"]
                if rc["bob.db.replay.directory"]
                else "",
                extension=".mov",
            ),
        )

