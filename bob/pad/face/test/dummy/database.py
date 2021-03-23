from bob.bio.base.test.utils import atnt_database_directory
import bob.io.base
import os
from bob.pipelines import DelayedSample
from bob.pad.base.pipelines.vanilla_pad.abstract_classes import Database
from bob.db.base.utils import check_parameters_for_validity, convert_names_to_lowlevel
from bob.bio.video import VideoLikeContainer


def DummyPadSample(
    path, original_directory, client_id, key, attack_type, none_annotations=False
):
    def load():
        file_name = os.path.join(original_directory, path + ".pgm")
        data = bob.io.base.load(file_name)[None, ...]
        indices = [os.path.basename(file_name)]
        fc = VideoLikeContainer(data, indices)
        return fc

    annotations = None
    if not none_annotations:
        annotations = {"0": {"topleft": (0, 0), "bottomright": (112, 92)}}

    return DelayedSample(
        load,
        client_id=client_id,
        key=key,
        attack_type=attack_type,
        is_bonafide=attack_type is None,
        annotations=annotations,
    )


class DummyDatabase(Database):
    def __init__(self, none_annotations=False):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__()
        self.original_directory = atnt_database_directory()
        import bob.db.atnt

        self._db = bob.db.atnt.Database()
        self.low_level_names = ("world", "dev")
        self.high_level_names = ("train", "dev")
        self.none_annotations = none_annotations

    def _make_bio(self, files):
        files = [
            DummyPadSample(
                path=f.path,
                original_directory=self.original_directory,
                client_id=f.client_id,
                key=f.id,
                attack_type=None,
                none_annotations=self.none_annotations,
            )
            for f in files
        ]
        return files

    def samples(self, groups=None, protocol=None, purposes=None, **kwargs):
        groups = check_parameters_for_validity(
            groups, "groups", self.high_level_names, default_parameters=None
        )
        groups = convert_names_to_lowlevel(
            groups, self.low_level_names, self.high_level_names
        )
        purposes = list(
            check_parameters_for_validity(
                purposes,
                "purposes",
                ("real", "attack"),
                default_parameters=("real", "attack"),
            )
        )
        if "real" in purposes:
            purposes.remove("real")
            purposes.append("enroll")
        if "attack" in purposes:
            purposes.remove("attack")
            purposes.append("probe")
        return self._make_bio(
            self._db.objects(None, groups, purposes, protocol, **kwargs)
        )

    def fit_samples(self):
        return self.samples(groups="train")

    def predict_samples(self, group="dev"):
        return self.samples(groups=group)


database = DummyDatabase()
