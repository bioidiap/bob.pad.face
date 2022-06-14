from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import bob.pipelines as mario

from bob.pad.face.transformer import VideoToFrames

preprocessor = globals()["preprocessor"]
extractor = globals()["extractor"]

# Classifier #
param_grid = [
    {
        "C": [2**P for P in range(-3, 14, 2)],
        "gamma": [2**P for P in range(-15, 0, 2)],
        "kernel": ["rbf"],
    },
]


# TODO: The grid search below does not take into account splitting frames of
# each video into a separate group. You might have frames of the same video in
# both groups of training and validation.

# TODO: This gridsearch can also be part of dask graph using dask-ml and the
# ``bob_fit_supports_dask_array`` tag from bob.pipelines.
classifier = GridSearchCV(SVC(), param_grid=param_grid, cv=3)
classifier = mario.wrap(
    ["sample"],
    classifier,
    fit_extra_arguments=[("y", "is_bonafide")],
)

# Pipeline #
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("extractor", extractor),
        ("video_to_frames", VideoToFrames()),
        ("svm", classifier),
    ]
)
