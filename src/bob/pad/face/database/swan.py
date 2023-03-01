import logging

from clapper.rc import UserDefaults

from bob.bio.base.database.utils import download_file
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample

logger = logging.getLogger(__name__)
rc = UserDefaults("bobrc.toml")


def SwanPadDatabase(
    protocol="pad_p2_face_f1",
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    annotation_directory=None,
    annotation_type=None,
    fixed_positions=None,
    **kwargs,
):
    name = "pad-face-swan-ce83ebd8.tar.gz"
    dataset_protocols_path = download_file(
        urls=[f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        destination_filename=name,
        destination_sub_directory="protocols",
        checksum="ce83ebd8",
    )

    if annotation_directory is None:
        name = "annotations-swan-mtcnn-9f9e12d8.tar.gz"
        annotation_directory = download_file(
            urls=[
                f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"
            ],
            destination_filename=name,
            destination_sub_directory="annotations",
            checksum="9f9e12d8",
        )
        annotation_type = "eyes-center"

    transformer = VideoPadSample(
        original_directory=rc.get("bob.db.swan.directory"),
        annotation_directory=annotation_directory,
        selection_style=selection_style,
        max_number_of_frames=max_number_of_frames,
        step_size=step_size,
        keep_extension_for_annotation=True,
    )

    database = FileListPadDatabase(
        name="swan",
        dataset_protocols_path=dataset_protocols_path,
        protocol=protocol,
        transformer=transformer,
        **kwargs,
    )
    database.annotation_type = annotation_type
    database.fixed_positions = fixed_positions
    return database
