import logging

from clapper.rc import UserDefaults

from bob.bio.base.database.utils import download_file
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample

logger = logging.getLogger(__name__)
rc = UserDefaults("bobrc.toml")


def CasiaFasdPadDatabase(
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    annotation_directory=None,
    annotation_type=None,
    fixed_positions=None,
    **kwargs,
):
    name = "pad-face-casia-fasd-0b07ea45.tar.gz"
    dataset_protocols_path = download_file(
        urls=[f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        destination_filename=name,
        destination_sub_directory="protocols/pad",
        checksum="0b07ea45",
    )

    transformer = VideoPadSample(
        original_directory=rc.get("bob.db.casia_fasd.directory"),
        annotation_directory=annotation_directory,
        selection_style=selection_style,
        max_number_of_frames=max_number_of_frames,
        step_size=step_size,
    )

    database = FileListPadDatabase(
        name="casia-fsd",
        dataset_protocols_path=dataset_protocols_path,
        protocol="grandtest",
        transformer=transformer,
        **kwargs,
    )
    database.annotation_type = annotation_type
    database.fixed_positions = fixed_positions
    return database
