from bob.io.video import reader
from bob.ip.base import scale, block, block_output_shape
from bob.ip.facedetect import bounding_box_from_annotation
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
    for frame in video:
        yield frame


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
    frames = paddb.frames(padfile)
    for image in frames:
        yield image


def normalize_detections(detections, nframes, max_age=-1, faceSizeFilter=0):
    """Calculates a list of "nframes" with the best possible detections taking
    into consideration the ages of the last valid detection on the detections
    list.

    Parameters
    ----------
    detections : dict
        A dictionary containing keys that indicate the frame number of the
        detection and a value which is a BoundingBox object.

    nframes : int
        An integer indicating how many frames has the video that will be
        analyzed.

    max_age : :obj:`int`, optional
        An integer indicating for a how many frames a detected face is valid if
        no detection occurs after such frame. A value of -1 == forever

    faceSizeFilter : :obj:`int`, optional
        The minimum required size of face height (in pixels)

    Yields
    ------
    object
        The bounding box or None.
    """
    curr = None
    age = 0

    for k in range(nframes):
        if detections and k in detections and \
                (detections[k].size[0] > faceSizeFilter):
            curr = detections[k]
            age = 0
        elif max_age < 0 or age < max_age:
            age += 1
        else:  # no detections and age is larger than maximum allowed
            curr = None

        yield curr


def yield_faces(database, padfile, **kwargs):
    """Yields face images of a padfile. It uses the annotations from the
    database. The annotations are further normalized.

    Parameters
    ----------
    database : :any:`bob.pad.base.database.PadDatabase`
        A face PAD database. This database needs to have implemented the
        `frames` method.
    padfile : :any:`bob.pad.base.database.PadFile`
        The padfile to return the faces.
    **kwargs
        They are passed to :any:`normalize_detections`.

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
    nframes = database.number_of_frames(padfile)
    # read annotation
    annots = database.annotations(padfile)
    if annots is None:
        raise ValueError("No annotations were returned.")
    # normalize annotations
    annots = {int(k): bounding_box_from_annotation(**v)
              for k, v in six.iteritems(annots)}
    bounding_boxes = normalize_detections(annots, nframes, **kwargs)
    for frame, bbx in six.moves.zip(frames_gen, bounding_boxes):
        if bbx is None:
            continue
        face = frame[..., bbx.top:bbx.bottom, bbx.left:bbx.right]
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
