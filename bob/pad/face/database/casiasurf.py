import logging
import os
from functools import partial

import bob.io.base
from bob.bio.video import VideoLikeContainer
from bob.extension import rc
from bob.extension.download import get_file
from bob.pad.base.database import FileListPadDatabase
from bob.pipelines import DelayedSample
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)


def load_multi_stream(mods, paths):
    retval = {}
    for mod, path in zip(mods, paths):
        data = bob.io.base.load(path)
        fc = VideoLikeContainer(data, [0])
        retval[mod] = fc

    if len(retval) == 1:
        retval = retval[mods[0]]

    return retval


def casia_surf_multistream_load(samples, original_directory, stream_type):
    mod_to_attr = {}
    mod_to_attr["color"] = "filename"
    mod_to_attr["infrared"] = "ir_filename"
    mod_to_attr["depth"] = "depth_filename"

    mods = []
    if isinstance(stream_type, str) and stream_type != "all":
        mods = [stream_type]
    elif isinstance(stream_type, str) and stream_type == "all":
        mods = ["color", "infrared", "depth"]
    else:
        for m in stream_type:
            mods.append(m)

    def _load(sample):
        paths = []
        for mod in mods:
            paths.append(
                os.path.join(original_directory or "", getattr(sample, mod_to_attr[mod]))
            )
        data = partial(load_multi_stream, mods, paths)
        return DelayedSample(data, parent=sample, annotations=None)

    return [_load(s) for s in samples]


def CasiaSurfMultiStreamSample(original_directory, stream_type):
    return FunctionTransformer(
        casia_surf_multistream_load,
        kw_args=dict(original_directory=original_directory, stream_type=stream_type),
    )


def CasiaSurfPadDatabase(
    stream_type="all",
    **kwargs,
):
    """The CASIA SURF Face PAD database interface.

    Parameters
    ----------
    stream_type : str
        A str or a list of str of the following choices: ``all``, ``color``, ``depth``, ``infrared``, by default ``all``

    The returned sample either have their data as a VideoLikeContainer or
    a dict of VideoLikeContainers depending on the chosen stream_type.
    """
    name = "pad-face-casia-surf-252f86f2.tar.gz"
    dataset_protocols_path = get_file(
        name,
        [f"http://www.idiap.ch/software/bob/data/bob/bob.pad.face/{name}"],
        cache_subdir="protocols",
        file_hash="252f86f2",
    )

    transformer = CasiaSurfMultiStreamSample(
        original_directory=rc.get("bob.db.casiasurf.directory"),
        stream_type=stream_type,
    )

    database = FileListPadDatabase(
        dataset_protocols_path,
        protocol="all",
        transformer=transformer,
        **kwargs,
    )
    database.annotation_type = None
    database.fixed_positions = None
    return database
