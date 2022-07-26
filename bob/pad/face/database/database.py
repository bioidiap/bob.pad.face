import os

from functools import partial

from sklearn.preprocessing import FunctionTransformer

from bob.bio.base.utils.annotations import read_annotation_file
from bob.bio.video import VideoAsArray
from bob.pipelines import DelayedSample


def get_no_transform(x):
    return None


def delayed_video_load(
    samples,
    original_directory,
    annotation_directory=None,
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    get_transform=None,
    keep_extension_for_annotation=False,
):
    get_transform = get_transform or get_no_transform

    original_directory = original_directory or ""
    annotation_directory = annotation_directory or ""
    results = []
    for sample in samples:
        video_path = os.path.join(original_directory, sample.filename)
        data = partial(
            VideoAsArray,
            path=video_path,
            selection_style=selection_style,
            max_number_of_frames=max_number_of_frames,
            step_size=step_size,
            transform=get_transform(sample),
        )
        delayed_attributes = None
        if annotation_directory:
            path = sample.filename
            if not keep_extension_for_annotation:
                path = os.path.splitext(sample.filename)[0]
            delayed_annotations = partial(
                read_annotation_file,
                file_name=f"{annotation_directory}:{path}.json",
                annotation_type="json",
            )
            delayed_attributes = {"annotations": delayed_annotations}
        if sample.attack_type == "":
            sample.attack_type = None
        sample.is_bonafide = sample.attack_type is None
        if not hasattr(sample, "key"):
            sample.key = sample.filename

        results.append(
            DelayedSample(
                data,
                parent=sample,
                delayed_attributes=delayed_attributes,
            )
        )
    return results


def VideoPadSample(
    original_directory,
    annotation_directory=None,
    selection_style=None,
    max_number_of_frames=None,
    step_size=None,
    get_transform=None,
    keep_extension_for_annotation=False,
):
    return FunctionTransformer(
        delayed_video_load,
        validate=False,
        kw_args=dict(
            original_directory=original_directory,
            annotation_directory=annotation_directory,
            selection_style=selection_style,
            max_number_of_frames=max_number_of_frames,
            step_size=step_size,
            get_transform=get_transform,
            keep_extension_for_annotation=keep_extension_for_annotation,
        ),
    )
