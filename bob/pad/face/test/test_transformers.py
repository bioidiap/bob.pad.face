#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
"""Test Units
"""
# ==============================================================================
# Import what is needed here:
import numpy as np

from bob.io.base.test_utils import datafile

from bob.io.base import load

import bob.bio.video

from bob.bio.face.color import rgb_to_gray

from bob.pad.face.extractor import ImageQualityMeasure


def notest_video_quality_measure():
    """
    Test ImageQualityMeasure with Wrapper extractor.
    """

    image = load(datafile("test_image.png", "bob.pad.face.test"))
    annotations = {"topleft": (95, 155), "bottomright": (215, 265)}

    video, annotations = convert_image_to_video_data(image, annotations, 2)

    GALBALLY = True
    MSU = True
    DTYPE = None

    extractor = bob.bio.video.extractor.Wrapper(
        ImageQualityMeasure(galbally=GALBALLY, msu=MSU, dtype=DTYPE)
    )

    features = extractor(video)

    assert len(features) == 2
    assert len(features[0][1]) == 139
    assert (features[0][1] == features[-1][1]).all()
    assert (features[0][1][0] - 2.7748559659812599e-05) < 0.000001
    assert (features[0][1][-1] - 0.16410418866596271) < 0.000001


def test_video_to_frames():
    # create a list of samples of videos
    # make sure some frames are None
    # call transform and check if the None frames are dropped
    import bob.pipelines as mario
    from bob.bio.video import VideoLikeContainer
    from bob.pad.face.transformer import VideoToFrames

    videos = [[0, 1, 2, None, 3], [None, None, None]]
    video_container = [VideoLikeContainer(v, range(len(v))) for v in videos]
    samples = [mario.Sample(v, key=i) for i, v in enumerate(video_container)]
    frame_samples = VideoToFrames().transform(samples)
    assert len(frame_samples) == 4
    assert all(s.key == 0 for s in frame_samples)
    assert [s.data for s in frame_samples] == [0, 1, 2, 3]
    assert [s.frame_id for s in frame_samples] == [0, 1, 2, 4]
