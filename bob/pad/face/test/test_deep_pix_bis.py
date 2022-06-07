import pkg_resources

import bob.io.base as io
import bob.pipelines as mario

from bob.bio.video import VideoLikeContainer


def _sample_video():
    path = pkg_resources.resource_filename(
        "bob.pad.face", "test/data/test_image.png"
    )
    img = io.load(path)
    video = VideoLikeContainer(img, [0])
    sample = mario.Sample(video, key="sample", annotations=None)
    return sample


def test_pipeline():
    from bob.pad.face.config.deep_pix_bis import pipeline

    sample = _sample_video()
    prediction = pipeline.predict_proba([sample])[0]
    assert prediction.data < 0.04
