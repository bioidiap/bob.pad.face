import logging

from bob.extension import rc
from bob.extension.download import get_file
from bob.pad.base.database import FileListPadDatabase
from bob.pad.face.database import VideoPadSample

logger = logging.getLogger(__name__)


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
    name = "pad-face-swan-711dffcf.tar.gz"
    dataset_protocols_path = get_file(
        name,
        [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        cache_subdir="protocols",
        file_hash="711dffcf",
    )

    if annotation_directory is None:
        name = "annotations-swan-mtcnn-cff2f062.tar.xz"
        annotation_directory = get_file(
            name,
            [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
            cache_subdir="annotations",
            file_hash="cff2f062",
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
        dataset_protocols_path,
        protocol,
        transformer=transformer,
        **kwargs,
    )
    database.annotation_type = annotation_type
    database.fixed_positions = fixed_positions
    return database
