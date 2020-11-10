# Legacy imports
from bob.bio.face.helpers import face_crop_solver
from bob.bio.video import VideoLikeContainer
from bob.bio.video.transformer import Wrapper as TransformerWrapper
from bob.pad.face.extractor import ImageQualityMeasure

# new imports
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from bob.pad.face.transformer import VideoToFrames
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
    save_func=VideoLikeContainer.save,
    load_func=VideoLikeContainer.load,
)

# Legacy extractor #
extractor = TransformerWrapper(ImageQualityMeasure(galbally=True, msu=True, dtype=None))
extractor = mario.wrap(
    ["sample", "checkpoint"],
    extractor,
    features_dir="temp/iqm-features",
    save_func=VideoLikeContainer.save,
    load_func=VideoLikeContainer.load,
)

# new stuff #
frame_cont_to_array = VideoToFrames()

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
frames_classifier = make_pipeline(frame_cont_to_array, classifier)
pipeline = make_pipeline(preprocessor, extractor, frames_classifier)
