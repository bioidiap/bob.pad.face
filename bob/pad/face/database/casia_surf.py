import logging
import os

from functools import partial

from sklearn.preprocessing import FunctionTransformer

import bob.io.base

from bob.bio.video import VideoLikeContainer
from bob.extension import rc
from bob.pad.base.database import FileListPadDatabase
from bob.pipelines import CSVToSamples, DelayedSample

logger = logging.getLogger(__name__)


def load_multi_stream(path):
    data = bob.io.base.load(path)
    video = VideoLikeContainer(data[None, ...], [0])
    return video


def casia_surf_multistream_load(samples, original_directory):
    mod_to_attr = {}
    mod_to_attr["color"] = "filename"
    mod_to_attr["infrared"] = "ir_filename"
    mod_to_attr["depth"] = "depth_filename"
    mods = list(mod_to_attr.keys())

    def _load(sample):
        paths = dict()
        for mod in mods:
            paths[mod] = os.path.join(
                original_directory or "", getattr(sample, mod_to_attr[mod])
            )
        data = partial(load_multi_stream, paths["color"])
        depth = partial(load_multi_stream, paths["depth"])
        infrared = partial(load_multi_stream, paths["infrared"])
        subject = None
        key = sample.filename
        is_bonafide = sample.is_bonafide == "1"
        attack_type = None if is_bonafide else "attack"

        return DelayedSample(
            data,
            parent=sample,
            subject=subject,
            key=key,
            attack_type=attack_type,
            is_bonafide=is_bonafide,
            annotations=None,
            delayed_attributes={"depth": depth, "infrared": infrared},
        )

    return [_load(s) for s in samples]


def CasiaSurfMultiStreamSample(original_directory):
    return FunctionTransformer(
        casia_surf_multistream_load,
        kw_args=dict(original_directory=original_directory),
    )


class CasiaSurfPadDatabase(FileListPadDatabase):
    """The CASIA SURF Face PAD database interface.

    Parameters
    ----------
    stream_type : str
        A str or a list of str of the following choices: ``all``, ``color``, ``depth``, ``infrared``, by default ``all``

    The returned sample either have their data as a VideoLikeContainer or
    a dict of VideoLikeContainers depending on the chosen stream_type.
    """

    def __init__(
        self,
        **kwargs,
    ):
        original_directory = rc.get("bob.db.casia_surf.directory")
        if original_directory is None or not os.path.isdir(original_directory):
            raise FileNotFoundError(
                "The original_directory is not set. Please set it in the terminal using `bob config set bob.db.casia_surf.directory /path/to/database/CASIA-SURF/`."
            )
        transformer = CasiaSurfMultiStreamSample(
            original_directory=original_directory,
        )
        super().__init__(
            dataset_protocols_path=original_directory,
            protocol="all",
            reader_cls=partial(
                CSVToSamples,
                dict_reader_kwargs=dict(
                    delimiter=" ",
                    fieldnames=[
                        "filename",
                        "ir_filename",
                        "depth_filename",
                        "is_bonafide",
                    ],
                ),
            ),
            transformer=transformer,
            **kwargs,
        )
        self.annotation_type = None
        self.fixed_positions = None

    def protocols(self):
        return ["all"]

    def groups(self):
        return ["train", "dev", "eval"]

    def list_file(self, group):
        filename = {
            "train": "train_list.txt",
            "dev": "val_private_list.txt",
            "eval": "test_private_list.txt",
        }[group]
        return os.path.join(self.dataset_protocols_path, filename)
