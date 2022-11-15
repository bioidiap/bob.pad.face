import logging
import os

from functools import partial

import h5py
import numpy as np

from sklearn.preprocessing import FunctionTransformer

from bob.bio.video import VideoLikeContainer, select_frames
from bob.extension import rc
from bob.extension.download import get_file
from bob.pad.base.database import FileListPadDatabase
from bob.pipelines import DelayedSample

logger = logging.getLogger(__name__)


def load_frames_from_hdf5(
    hdf5_file,
    key="Color_Data",
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
):
    with h5py.File(hdf5_file) as f:
        video = f[key][()]
        # reduce the shape of depth from (N, C, H, W) to (N, H, W) since H == 1
        video = np.squeeze(video)

    indices = select_frames(
        len(video),
        max_number_of_frames=max_number_of_frames,
        selection_style=selection_style,
        step_size=step_size,
    )
    data = VideoLikeContainer(video[indices], indices)

    return data


def load_annotations_from_hdf5(
    hdf5_file,
):
    with h5py.File(hdf5_file) as f:
        eye_pos = f["Eye_Pos"][()]

    annotations = {
        str(i): {
            "reye": [row[1], row[0]],
            "leye": [row[3], row[2]],
        }
        for i, row in enumerate(eye_pos)
    }
    return annotations


def delayed_maskattack_video_load(
    samples,
    original_directory,
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
):

    original_directory = original_directory or ""
    results = []
    for sample in samples:
        hdf5_file = os.path.join(original_directory, sample.filename)
        data = partial(
            load_frames_from_hdf5,
            key="Color_Data",
            hdf5_file=hdf5_file,
            selection_style=selection_style,
            max_number_of_frames=max_number_of_frames,
            step_size=step_size,
        )
        depth = partial(
            load_frames_from_hdf5,
            key="Depth_Data",
            hdf5_file=hdf5_file,
            selection_style=selection_style,
            max_number_of_frames=max_number_of_frames,
            step_size=step_size,
        )
        annotations = partial(
            load_annotations_from_hdf5,
            hdf5_file=hdf5_file,
        )
        delayed_attributes = {
            "annotations": annotations,
            "depth": depth,
        }

        results.append(
            DelayedSample(
                data,
                parent=sample,
                delayed_attributes=delayed_attributes,
            )
        )
    return results


def MaskAttackPadSample(
    original_directory,
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
):
    return FunctionTransformer(
        delayed_maskattack_video_load,
        validate=False,
        kw_args=dict(
            original_directory=original_directory,
            selection_style=selection_style,
            max_number_of_frames=max_number_of_frames,
            step_size=step_size,
        ),
    )


def MaskAttackPadDatabase(
    protocol="classification",
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    **kwargs,
):
    name = "pad-face-mask-attack-2ab2032c.tar.gz"
    dataset_protocols_path = get_file(
        name,
        [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        cache_subdir="protocols",
        file_hash="2ab2032c",
    )

    transformer = MaskAttackPadSample(
        original_directory=rc.get("bob.db.mask_attack.directory"),
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
    database.annotation_type = "eyes-center"
    database.fixed_positions = None
    return database
