# Legacy imports
from bob.bio.face.helpers import face_crop_solver
from bob.bio.video import FrameSelector
from bob.bio.video.extractor import Wrapper as ExtractorWrapper
from bob.bio.video.transformer import Wrapper as TransformerWrapper
from bob.pad.face.extractor import ImageQualityMeasure

# new imports
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from bob.pad.base.pipelines.vanilla_pad import FrameContainersToFrames
from bob.bio.base.wrappers import wrap_sample_extractor
import bob.pipelines as mario

database = globals().get("database")
if database is not None:
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None

# Preprocessor #
cropper = face_crop_solver(cropped_image_size=64, cropped_positions=annotation_type)
preprocessor = TransformerWrapper(cropper)
preprocessor = mario.wrap(
    ["sample", "checkpoint"],
    preprocessor,
    transform_extra_arguments=(("annotations", "annotations"),),
    features_dir="temp/faces-64",
)

# Legacy extractor #
extractor = TransformerWrapper(ImageQualityMeasure(galbally=True, msu=True, dtype=None))
extractor = mario.wrap(
    ["sample", "checkpoint"],
    extractor,
    features_dir="temp/iqm-features",
)

# new stuff #
frame_cont_to_array = FrameContainersToFrames()

param_grid = [
    {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]},
]
classifier = GridSearchCV(SVC(), param_grid=param_grid, cv=3)
classifier = mario.wrap(
    ["sample", "checkpoint"],
    classifier,
    fit_extra_arguments=[("y", "is_bonafide")],
    model_path="temp/svm.pkl",
)


# pipeline #
# stateless_pipeline = mario.transformers.StatelessPipeline(
#     [
#         ("preprocessor", preprocessor),
#         ("extractor", extractor),
#         ("frame_cont_to_array", frame_cont_to_array),
#     ]
# )
pipeline = make_pipeline(preprocessor, extractor, frame_cont_to_array, classifier)
