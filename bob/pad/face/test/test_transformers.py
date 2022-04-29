import bob.pipelines as mario

from bob.bio.video import VideoLikeContainer
from bob.pad.face.transformer import VideoToFrames


def test_video_to_frames():
    # create a list of samples of videos
    # make sure some frames are None
    # call transform and check if the None frames are dropped

    videos = [[0, 1, 2, None, 3], [None, None, None]]
    video_container = [VideoLikeContainer(v, range(len(v))) for v in videos]
    samples = [mario.Sample(v, key=i) for i, v in enumerate(video_container)]
    frame_samples = VideoToFrames().transform(samples)
    assert len(frame_samples) == 4
    assert all(s.key == 0 for s in frame_samples)
    assert [s.data for s in frame_samples] == [0, 1, 2, 3]
    assert [s.frame_id for s in frame_samples] == [0, 1, 2, 4]
