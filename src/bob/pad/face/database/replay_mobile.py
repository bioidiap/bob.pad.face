import logging

from clapper.rc import UserDefaults
from sklearn.pipeline import make_pipeline

from bob.bio.base.database.utils import download_file
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample
from bob.pipelines.transformers import Str_To_Types, str_to_bool

logger = logging.getLogger(__name__)
rc = UserDefaults("bobrc.toml")


def get_rm_video_transform(sample):
    should_flip = sample.should_flip

    def transform(video):
        if not should_flip:
            # after changing to imageio-ffmpeg, we need to flip other way around
            video = video[..., ::-1]
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
    name = "pad-face-replay-mobile-620dded2.tar.gz"
    dataset_protocols_path = download_file(
        urls=[f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        destination_filename=name,
        destination_sub_directory="protocols/pad",
        checksum="620dded2",
    )

    if annotation_directory is None:
        name = "annotations-replay-mobile-mtcnn-20055a07.tar.gz"
        annotation_directory = download_file(
            urls=[
                f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"
            ],
            destination_filename=name,
            destination_sub_directory="annotations",
            checksum="20055a07",
        )
        annotation_type = "eyes-center"

    transformer = make_pipeline(
        Str_To_Types(fieldtypes=dict(should_flip=str_to_bool)),
        VideoPadSample(
            original_directory=rc.get("bob.db.replay_mobile.directory"),
            annotation_directory=annotation_directory,
            selection_style=selection_style,
            max_number_of_frames=max_number_of_frames,
            step_size=step_size,
            get_transform=get_rm_video_transform,
        ),
    )

    database = FileListPadDatabase(
        name="replay-mobile",
        dataset_protocols_path=dataset_protocols_path,
        protocol=protocol,
        transformer=transformer,
        **kwargs,
    )
    database.annotation_type = annotation_type
    database.fixed_positions = fixed_positions
    return database
