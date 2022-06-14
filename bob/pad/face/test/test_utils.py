import imageio
import numpy
import pytest

from bob.pad.face.test.dummy.database import DummyDatabase as Database
from bob.pad.face.utils import (
    blocks,
    frames,
    number_of_frames,
    scale_face,
    yield_faces,
)


def get_pad_sample(none_annotations=False):
    sample = Database(none_annotations=none_annotations).samples(
        ("train", "dev")
    )[0]
    return sample


image = get_pad_sample().data[0]


def test_video_frames():
    # get the path to cockatoo.mp4 from imageio-ffmpeg
    path = imageio.core.Request(
        "imageio:cockatoo.mp4", "r"
    ).get_local_filename()
    # read 2 frames
    for i, frame in enumerate(frames(path)):
        assert frame.shape == (3, 720, 1280), frame.shape
        assert frame.ndim == 3, frame.ndim
        if i == 0:
            numpy.testing.assert_equal(frame[:, 0, 0], [116, 119, 104])
        elif i == 1:
            numpy.testing.assert_equal(frame[:, 0, 0], [116, 119, 104])
        else:
            break

    # test number of frames
    n_frames = number_of_frames(path)
    assert n_frames == 280, n_frames


def dummy_cropper(frame, annotations=None):
    return frame


def test_yield_frames():
    padfile = get_pad_sample()
    nframes = len(padfile.data)
    assert nframes == 1, nframes
    for frame in padfile.data:
        assert frame.ndim == 2
        assert frame.shape == (112, 92)


def test_yield_faces_1():
    with pytest.raises(ValueError):
        padfile = get_pad_sample(none_annotations=True)
        for face in yield_faces(padfile, dummy_cropper):
            pass


def test_yield_faces_2():
    padfile = get_pad_sample(none_annotations=False)
    assert len(list(yield_faces(padfile, dummy_cropper)))
    for face in yield_faces(padfile, dummy_cropper):
        assert face.ndim == 2
        assert face.shape == padfile.data.shape[1:]


def test_scale_face():
    # gray-scale image
    face = image
    scaled_face = scale_face(face, 64)
    assert scaled_face.dtype == "float64"
    assert scaled_face.shape == (64, 64)
    # color image
    scaled_face = scale_face(numpy.array([face, face, face]), 64)
    assert scaled_face.dtype == "float64"
    assert scaled_face.shape == (3, 64, 64)
    assert (scaled_face[0] == scaled_face[1]).all()
    assert (scaled_face[0] == scaled_face[2]).all()


def test_blocks():
    # gray-scale image
    patches = blocks(image, (28, 28))
    assert patches.shape == (12, 28, 28), patches.shape
    # color image
    patches_gray = patches
    patches = blocks([image, image, image], (28, 28))
    assert patches.shape == (12, 3, 28, 28), patches.shape
    assert (patches_gray == patches[:, 0, ...]).all()
    assert (patches_gray == patches[:, 1, ...]).all()
    assert (patches_gray == patches[:, 2, ...]).all()
    # color video
    patches = blocks([[image, image, image]], (28, 28))
    assert patches.shape == (12, 3, 28, 28), patches.shape
    assert (patches_gray == patches[:, 0, ...]).all()
    assert (patches_gray == patches[:, 1, ...]).all()
    assert (patches_gray == patches[:, 2, ...]).all()


def test_block_raises1():
    with pytest.raises(ValueError):
        blocks(image[0], (28, 28))


def test_block_raises2():
    with pytest.raises(ValueError):
        blocks([[[image]]], (28, 28))
