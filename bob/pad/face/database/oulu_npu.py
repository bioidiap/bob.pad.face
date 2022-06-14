import logging

from bob.extension import rc
from bob.extension.download import get_file
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample

logger = logging.getLogger(__name__)


def OuluNpuPadDatabase(
    protocol="Protocol_1",
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    annotation_directory=None,
    annotation_type=None,
    fixed_positions=None,
    **kwargs,
):
    name = "pad-face-oulunpu-75221078.tar.gz"
    dataset_protocols_path = get_file(
        name,
        [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        cache_subdir="protocols",
        file_hash="75221078",
    )

    if annotation_directory is None:
        name = "annotations-oulunpu-mtcnn-3bee9b6d.tar.xz"
        annotation_directory = get_file(
            name,
            [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
            cache_subdir="annotations",
            file_hash="3bee9b6d",
        )
        annotation_type = "eyes-center"

    transformer = VideoPadSample(
        original_directory=rc.get("bob.db.oulu_npu.directory"),
        annotation_directory=annotation_directory,
        selection_style=selection_style,
        max_number_of_frames=max_number_of_frames,
        step_size=step_size,
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
