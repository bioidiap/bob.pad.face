from bob.bio.face.annotator import min_face_size_validator
from bob.bio.video.annotator import normalize_annotations
from bob.io.video import reader
from bob.ip.base import scale, block, block_output_shape
from bob.ip.color import rgb_to_yuv, rgb_to_hsv
from bob.ip.facedetect import bounding_box_from_annotation
from functools import partial
import numpy
import six


def frames(path):
    """Yields the frames of a video file.

    Parameters
    ----------
    path : str
        Path to the video file.

    Yields
    ------
    :any:`numpy.array`
        A frame of the video. The size is (3, 240, 320).
    """
    video = reader(path)
    return iter(video)


def number_of_frames(path):
    """returns the number of frames of a video file.

    Parameters
    ----------
    path : str
        Path to the video file.

    Returns
    -------
    int
        The number of frames. Then, it yields the frames.
    """
    video = reader(path)
    return video.number_of_frames


def yield_frames(paddb, padfile):
    """Loads the frames of a video PAD database.

    Parameters
    ----------
    paddb : :any:`bob.pad.base.database.PadDatabase`
        The video PAD database. The database needs to have implemented the
        `.frames()` method.
    padfile : :any:`bob.pad.face.database.VideoPadFile`
        The PAD file.

    Yields
    ------
    :any:`numpy.array`
        Frames of the PAD file one by one.
    """
    return paddb.frames(padfile)


def bbx_cropper(frame, annotations):
    bbx = bounding_box_from_annotation(**annotations)
    return frame[..., bbx.top:bbx.bottom, bbx.left:bbx.right]


def min_face_size_normalizer(annotations, max_age=15, **kwargs):
    return normalize_annotations(annotations,
                                 partial(min_face_size_validator, **kwargs),
                                 max_age=max_age)


def yield_faces(database, padfile, cropper, normalizer=None):
    """Yields face images of a padfile. It uses the annotations from the
    database. The annotations are further normalized.

    Parameters
    ----------
    database : :any:`bob.pad.base.database.PadDatabase`
        A face PAD database. This database needs to have implemented the
        `frames` method.
    padfile : :any:`bob.pad.base.database.PadFile`
        The padfile to return the faces.
    cropper : callable
        A face image cropper that works with database's annotations.
    normalizer : callable
        If not None, it should be a function that takes all the annotations of
        the whole video and yields normalized annotations frame by frame. It
        should yield same as ``annotations.items()``.

    Yields
    ------
    numpy.array
        Face images

    Raises
    ------
    ValueError
        If the database returns None for annotations.
    """
    frames_gen = database.frames(padfile)

    # read annotation
    annotations = database.annotations(padfile)
    if annotations is None:
        raise ValueError("No annotations were returned.")

    if normalizer is None:
        annotations_gen = annotations.items()
    else:
        annotations_gen = normalizer(annotations)

    # normalize annotations and crop faces
    for _, annot in annotations_gen:
        frame = six.next(frames_gen)
        if annot is None:
            continue
        face = cropper(frame, annotations=annot)
        if face is not None:
            yield face


def scale_face(face, face_height, face_width=None):
    """Scales a face image to the given size.

    Parameters
    ----------
    face : :any:`numpy.array`
        The face image. It can be 2D or 3D in bob image format.
    face_height : int
        The height of the scaled face.
    face_width : :obj:`None`, optional
        The width of the scaled face. If None, face_height is used.

    Returns
    -------
    :any:`numpy.array`
        The scaled face.
    """
    face_width = face_height if face_width is None else face_width
    shape = list(face.shape)
    shape[-2:] = (face_height, face_width)
    scaled_face = numpy.empty(shape, dtype='float64')
    scale(face, scaled_face)
    return scaled_face


def blocks(data, block_size, block_overlap=(0, 0)):
    """Extracts patches of an image

    Parameters
    ----------
    data : :any:`numpy.array`
        The image in gray-scale, color, or color video format.
    block_size : (int, int)
        The size of patches
    block_overlap : (:obj:`int`, :obj:`int`), optional
        The size of overlap of patches

    Returns
    -------
    :any:`numpy.array`
        The patches.

    Raises
    ------
    ValueError
        If data dimension is not between 2 and 4 (inclusive).
    """
    data = numpy.asarray(data)
    # if a gray scale image:
    if data.ndim == 2:
        output = block(data, block_size, block_overlap,
                       flat=True)
    # if a color image:
    elif data.ndim == 3:
        out_shape = list(data.shape[0:1]) + list(block_output_shape(
            data[0], block_size, block_overlap, flat=True))

        output = numpy.empty(out_shape, dtype=data.dtype)
        for i, img2d in enumerate(data):
            block(img2d, block_size, block_overlap, output[i], flat=True)
        output = numpy.moveaxis(output, 0, 1)
    # if a color video:
    elif data.ndim == 4:
        output = [blocks(img3d, block_size, block_overlap)
                  for img3d in data]
        output = numpy.concatenate(output, axis=0)
    else:
        raise ValueError("Unknown data dimension {}".format(data.ndim))
    return output


def color_augmentation(image, channels=('rgb',)):
    """Converts an RGB image to different color channels.

    Parameters
    ----------
    image : numpy.array
        The image in RGB Bob format.
    channels : :obj:`tuple`, optional
        List of channels to convert the image to. It can be any of ``rgb``,
        ``yuv``, ``hsv``.

    Returns
    -------
    numpy.array
        The image that contains several channels:
        ``(3*len(channels), height, width)``.
    """
    final_image = []

    if 'rgb' in channels:
        final_image.append(image)

    if 'yuv' in channels:
        final_image.append(rgb_to_yuv(image))

    if 'hsv' in channels:
        final_image.append(rgb_to_hsv(image))

    return numpy.concatenate(final_image, axis=0)


def _random_sample(A, size):
    return A[numpy.random.choice(A.shape[0], size, replace=False), ...]


def the_giant_video_loader(paddb, padfile,
                           region='whole', scaling_factor=None, cropper=None,
                           normalizer=None, patches=False,
                           block_size=(96, 96), block_overlap=(0, 0),
                           random_patches_per_frame=None, augment=None,
                           multiple_bonafide_patches=1):
    if region == 'whole':
        generator = yield_frames(paddb, padfile)
    elif region == 'crop':
        generator = yield_faces(
            paddb, padfile, cropper=cropper, normalizer=normalizer)
    else:
        raise ValueError("Invalid region value: `{}'".format(region))

    if scaling_factor is not None:
        generator = (scale(frame, scaling_factor)
                     for frame in generator)
    if patches:
        if random_patches_per_frame is None:
            generator = (
                patch for frame in generator
                for patch in blocks(frame, block_size, block_overlap))
        else:
            if padfile.attack_type is None:
                random_patches_per_frame *= multiple_bonafide_patches
            generator = (
                patch for frame in generator
                for patch in _random_sample(
                    blocks(frame, block_size, block_overlap),
                    random_patches_per_frame))

    if augment is not None:
        generator = (augment(frame) for frame in generator)

    return generator
