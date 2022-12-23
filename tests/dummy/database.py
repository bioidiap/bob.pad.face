import os

import bob.io.base

from bob.bio.video import VideoLikeContainer
from bob.pad.base.pipelines.abstract_classes import Database
from bob.pad.face.database import AtntPadDatabase
from bob.pipelines import DelayedSample


def DummyPadSample(
    path,
    original_directory,
    template_id,
    key,
    attack_type,
    none_annotations=False,
):
    def load():
        file_name = os.path.join(original_directory, f"{path}.pgm")
        data = bob.io.base.load(file_name)[None, ...]
        indices = [os.path.basename(file_name)]
        fc = VideoLikeContainer(data, indices)
        return fc

    annotations = None
    if not none_annotations:
        annotations = {"0": {"topleft": (0, 0), "bottomright": (112, 92)}}

    return DelayedSample(
        load,
        client_id=template_id,
        key=key,
        attack_type=attack_type,
        is_bonafide=attack_type is None,
        annotations=annotations,
    )


class DummyDatabase(Database):
    def __init__(self, none_annotations=False):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__()
        self._db = AtntPadDatabase()
        self.original_directory = self._db.dataset_original_directory
        self.none_annotations = none_annotations
        self.high_level_names = ["train", "dev", "eval"]
        self.low_level_names = ["dev", "eval"]

    def _make_bio(self, files):
        files = [
            DummyPadSample(
                path=f.path,
                original_directory=self.original_directory,
                template_id=f.template_id,
                key=f.id,
                attack_type=None,
                none_annotations=self.none_annotations,
            )
            for f in files
        ]
        return files

    def samples(self, groups=None, purposes=None, **kwargs):
        return self._make_bio(self._db.samples(groups))

    def fit_samples(self):
        return self.samples(groups="train")

    def predict_samples(self, group="dev"):
        return self.samples(groups=group)


database = DummyDatabase()
