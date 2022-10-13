import dask_ml.model_selection as dcv

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from bob.bio.face.annotator import MTCNN
from bob.bio.face.preprocessor import INormLBP
from bob.bio.face.utils import make_cropper, pad_default_cropping
from bob.pad.face.transformer import VideoToFrames
from bob.pad.face.transformer.histogram import SpatialHistogram
from bob.pipelines.wrappers import SampleWrapper


def _init_pipeline(database, crop_size=(112, 112), grid_size=(3, 3)):
    # Face Crop
    # --------------------------
    annotator = MTCNN(thresholds=(0.1, 0.2, 0.2))
    cropped_pos = pad_default_cropping(crop_size, database.annotation_type)
    cropper = make_cropper(
        cropped_image_size=crop_size,
        cropped_positions=cropped_pos,
        fixed_positions=database.fixed_positions,
        color_channel="rgb",
        annotator=annotator,
    )
    face_cropper = SampleWrapper(
        cropper[0], transform_extra_arguments=cropper[1], delayed_output=False
    )

    # Extract LBP
    # --------------------------
    lbp_extractor = INormLBP(face_cropper=None, color_channel="gray")
    lbp_extractor = SampleWrapper(lbp_extractor, delayed_output=False)

    # Histogram
    # --------------------------
    histo = SpatialHistogram(grid_size=grid_size, nbins=256)
    # histo = VideoWrapper(histo)
    histo = SampleWrapper(histo, delayed_output=False)

    # Classifier
    # --------------------------
    sk_classifier = SVC()
    param_grid = [
        {
            "C": [2**p for p in range(-3, 14, 2)],
            "gamma": [2**p for p in range(-15, 0, 2)],
            "kernel": ["rbf"],
        }
    ]
    cv = StratifiedGroupKFold(n_splits=3)
    sk_classifier = dcv.GridSearchCV(
        sk_classifier, param_grid=param_grid, cv=cv
    )
    fit_extra_arguments = [("y", "is_bonafide"), ("groups", "video_key")]
    classifier = SampleWrapper(
        sk_classifier,
        delayed_output=False,
        fit_extra_arguments=fit_extra_arguments,
    )

    # Full Pipeline
    # --------------------------
    return Pipeline(
        [
            ("video2frames", VideoToFrames()),
            ("cropper", face_cropper),
            ("lbp", lbp_extractor),
            ("spatial_histogram", histo),
            ("classifier", classifier),
        ]
    )


# Get database information, needed for face cropper
db = globals()["database"]
if db is None:
    raise ValueError("Missing database!")
# Pipeline #
pipeline = _init_pipeline(database=db)
