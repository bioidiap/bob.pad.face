from sklearn.pipeline import Pipeline

import bob.pipelines as mario

from bob.bio.face.preprocessor import FaceCrop
from bob.bio.face.utils import get_default_cropped_positions
from bob.bio.video.transformer import VideoWrapper
from bob.pad.face.deep_pix_bis import DeepPixBisClassifier
from bob.pad.face.transformer import VideoToFrames

database = globals().get("database")
if database is not None:
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None


# Preprocessor #
cropped_image_size = (224, 224)
cropped_positions = get_default_cropped_positions(
    "pad", cropped_image_size, annotation_type
)
cropper = FaceCrop(
    cropped_image_size=cropped_image_size,
    cropped_positions=cropped_positions,
    color_channel="rgb",
    fixed_positions=fixed_positions,
    dtype="uint8",
    annotator="mtcnn",
)

preprocessor = VideoWrapper(cropper)
preprocessor = mario.wrap(
    ["sample"],
    preprocessor,
    transform_extra_arguments=(("annotations", "annotations"),),
)

# Classifier #
classifier = DeepPixBisClassifier(model_file="oulunpu-p1")
classifier = mario.wrap(["sample"], classifier)


pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("video_to_frames", VideoToFrames()),
        ("classifier", classifier),
    ]
)
