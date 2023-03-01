import logging

from clapper.rc import UserDefaults

from bob.bio.base.database.utils import download_file
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample

logger = logging.getLogger(__name__)
rc = UserDefaults("bobrc.toml")


def ReplayAttackPadDatabase(
    protocol="grandtest",
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    annotation_directory=None,
    annotation_type=None,
    fixed_positions=None,
    **kwargs,
):
    name = "pad-face-replay-attack-aca6b46f.tar.gz"
    dataset_protocols_path = download_file(
        urls=[f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        destination_filename=name,
        destination_sub_directory="protocols/pad",
        checksum="aca6b46f",
    )

    if annotation_directory is None:
        name = "annotations-replay-attack-mtcnn-8d1f4c12.tar.gz"
        annotation_directory = download_file(
            urls=[
                f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"
            ],
            destination_filename=name,
            destination_sub_directory="annotations",
            checksum="8d1f4c12",
        )
        annotation_type = "eyes-center"

    transformer = VideoPadSample(
        original_directory=rc.get("bob.db.replay_attack.directory"),
        annotation_directory=annotation_directory,
        selection_style=selection_style,
        max_number_of_frames=max_number_of_frames,
        step_size=step_size,
    )

    database = FileListPadDatabase(
        name="replay-attack",
        dataset_protocols_path=dataset_protocols_path,
        protocol=protocol,
        transformer=transformer,
        **kwargs,
    )
    database.annotation_type = annotation_type
    database.fixed_positions = fixed_positions
    return database
