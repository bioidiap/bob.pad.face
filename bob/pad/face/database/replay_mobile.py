import logging
import os

import numpy as np
from bob.extension import rc
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample
from bob.pipelines.transformers import Str_To_Types, str_to_bool
from sklearn.pipeline import make_pipeline
from bob.extension.download import get_file

logger = logging.getLogger(__name__)


def get_rm_video_transform(sample):
    should_flip = sample.should_flip

    def transform(video):
        video = np.asarray(video)
        video = np.rollaxis(video, -1, -2)
        if should_flip:
            video = video[..., ::-1, :]
        return video

    return transform


def ReplayMobilePadDatabase(
    protocol="grandtest",
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    annotation_directory=None,
    annotation_type=None,
    fixed_positions=None,
    **kwargs,
):
    dataset_protocols_path = os.path.expanduser(
        "~/temp/bob_data/datasets/pad-face-replay-mobile.tar.gz"
    )
    if annotation_directory is None:
        annotation_directory = get_file(
            "replaymobile-mtcnn-annotations.tar.xz",
            [
                "http://www.idiap.ch/software/bob/data/bob/bob.db.replaymobile/master/replaymobile-mtcnn-annotations.tar.xz"
            ],
        )
        annotation_type = "eyes-center"
    transformer = make_pipeline(
        Str_To_Types(fieldtypes=dict(should_flip=str_to_bool)),
        VideoPadSample(
            original_directory=rc.get("bob.db.replaymobile.directory"),
            annotation_directory=annotation_directory,
            selection_style=selection_style,
            max_number_of_frames=max_number_of_frames,
            step_size=step_size,
            get_transform=get_rm_video_transform,
        ),
    )
    database = FileListPadDatabase(
        dataset_protocols_path,
        protocol,
        transformer=transformer,
        **kwargs,
    )
    database.annotation_type = annotation_type
    database.fixed_positions = fixed_positions
    return database
