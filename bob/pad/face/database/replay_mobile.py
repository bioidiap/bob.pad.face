import logging

import numpy as np
from bob.extension import rc
from bob.extension.download import get_file
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample
from bob.pipelines.transformers import Str_To_Types
from bob.pipelines.transformers import str_to_bool
from sklearn.pipeline import make_pipeline

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
    name = "pad-face-replay-mobile-586b7e81.tar.gz"
    dataset_protocols_path = get_file(
        name,
        [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        cache_subdir="protocols",
        file_hash="586b7e81",
    )

    if annotation_directory is None:
        name = "annotations-replaymobile-mtcnn-9cd6e452.tar.xz"
        annotation_directory = get_file(
            name,
            [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
            cache_subdir="annotations",
            file_hash="9cd6e452",
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
