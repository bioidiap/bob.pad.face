import bob.pipelines as mario
from bob.pad.face.transformer import VideoToFrames
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

preprocessor = globals().get("preprocessor")
extractor = globals().get("extractor")

# Classifier #
frame_cont_to_array = VideoToFrames()

param_grid = [
    {
        "C": [2 ** P for P in range(-3, 14, 2)],
        "gamma": [2 ** P for P in range(-15, 0, 2)],
        "kernel": ["rbf"],
    },
]

classifier = GridSearchCV(SVC(), param_grid=param_grid, cv=3)
classifier = mario.wrap(
    ["sample", "checkpoint"],
    classifier,
    fit_extra_arguments=[("y", "is_bonafide")],
    model_path="temp/svm.pkl",
)


# Pipeline #
frames_classifier = make_pipeline(frame_cont_to_array, classifier)
pipeline = make_pipeline(preprocessor, extractor, frames_classifier)
