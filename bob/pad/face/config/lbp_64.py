import bob.pipelines as mario
from bob.bio.face.helpers import face_crop_solver
from bob.bio.video.transformer import VideoWrapper
from bob.pad.face.extractor import LBPHistogram

database = globals().get("database")
if database is not None:
    annotation_type = database.annotation_type
    fixed_positions = database.fixed_positions
else:
    annotation_type = None
    fixed_positions = None

# Preprocessor #
cropper = face_crop_solver(
    cropped_image_size=64,
    cropped_positions=annotation_type,
    color_channel="gray",
    fixed_positions=fixed_positions,
)
preprocessor = VideoWrapper(cropper)
preprocessor = mario.wrap(
    ["sample"],
    preprocessor,
    transform_extra_arguments=(("annotations", "annotations"),),
)

# Extractor #
extractor = VideoWrapper(
    LBPHistogram(
        lbp_type="uniform",
        elbp_type="regular",
        radius=1,
        neighbors=8,
        circular=False,
        dtype=None,
    )
)
extractor = mario.wrap(["sample"], extractor)
