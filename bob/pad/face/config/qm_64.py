import bob.pipelines as mario
from bob.bio.face.helpers import face_crop_solver
from bob.bio.video.transformer import VideoWrapper
from bob.pad.face.extractor import ImageQualityMeasure

database = globals().get("database")
if database is not None:
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None

# Preprocessor #
cropper = face_crop_solver(cropped_image_size=64, cropped_positions=annotation_type)
preprocessor = VideoWrapper(cropper)
preprocessor = mario.wrap(
    ["sample"],
    preprocessor,
    transform_extra_arguments=(("annotations", "annotations"),),
)

# Extractor #
extractor = VideoWrapper(ImageQualityMeasure(galbally=True, msu=True, dtype=None))
extractor = mario.wrap(["sample"], extractor)
