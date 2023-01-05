import random

from collections import OrderedDict
from functools import partial

import numpy

from imageio import get_reader
from PIL import Image

import bob.io.image

from bob.bio.face.annotator import (
    bounding_box_from_annotation,
    min_face_size_validator,
)
from bob.bio.face.color import rgb_to_hsv, rgb_to_yuv
from bob.bio.video.annotator import normalize_annotations


def block(X, block_size, block_overlap, flat=False):
    """
    Parameters
    ----------
    X : numpy.ndarray
        The image to be split into blocks.
    block_size : tuple
        The size of the block.
    block_overlap : tuple
        The overlap of the block.

    Returns
    -------
    numpy.ndarray
        The image split into blocks.
    """

    assert len(block_size) == 2
    assert len(block_overlap) == 2

    size_ov_h = int(block_size[0] - block_overlap[0])
    size_ov_w = int(block_size[1] - block_overlap[1])
    n_blocks_h = int((X.shape[0] - block_overlap[0]) / size_ov_h)
    n_blocks_w = int((X.shape[1] - block_overlap[1]) / size_ov_w)

    blocks = numpy.zeros(shape=(n_blocks_h, n_blocks_w, size_ov_h, size_ov_w))
    for h in range(n_blocks_h):
        for w in range(n_blocks_w):

            blocks[h, w, :, :] = X[
                h * size_ov_h : h * size_ov_h + block_size[0],
                w * size_ov_w : w * size_ov_w + block_size[1],
            ]

    if flat:
        return blocks.reshape(
            n_blocks_h * n_blocks_w, blocks.shape[2], blocks.shape[3]
        )

    return blocks


def scale(image, scaling_factor):
    """
    Scales and image using PIL

    Parameters
    ----------

        image:
           Input image to be scaled

        new_shape: tuple
           Shape of the rescaled image



    """

    if isinstance(scaling_factor, float):
        new_size = tuple(
            (numpy.array(image.shape) * scaling_factor).astype(numpy.int)
        )
        return bob.io.image.to_bob(
            numpy.array(
                Image.fromarray(bob.io.image.to_matplotlib(image)).resize(
                    size=new_size
                ),
                dtype="float",
            )
        )

    elif isinstance(scaling_factor, tuple):

        if len(scaling_factor) > 2:
            scaling_factor = scaling_factor[1:]

        return bob.io.image.to_bob(
            numpy.array(
                Image.fromarray(bob.io.image.to_matplotlib(image)).resize(
                    size=scaling_factor
                ),
                dtype="float",
            )
        )
    else:
        raise ValueError(f"Scaling factor not supported: {scaling_factor}")


def frames(path):
    """Yields the frames of a video file.

    Parameters
    ----------
    path : str
        Path to the video file.

    Yields
    ------
    numpy.ndarray
        A frame of the video. The size is (3, 240, 320).
    """
    video = get_reader(path)
    for frame in video:
        yield bob.io.image.to_bob(frame)


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
    video = get_reader(path)
    return video.count_frames()


def bbx_cropper(frame, annotations):
    for source in ("direct", "eyes", None):
        try:
            bbx = bounding_box_from_annotation(source=source, **annotations)
            break
        except Exception:
            if source is None:
                raise
    return frame[..., bbx.top : bbx.bottom, bbx.left : bbx.right]


def min_face_size_normalizer(annotations, max_age=15, **kwargs):
    return normalize_annotations(
        annotations, partial(min_face_size_validator, **kwargs), max_age=max_age
    )


def yield_faces(pad_sample, cropper, normalizer=None):
    """Yields face images of a padfile. It uses the annotations from the
    database. The annotations are further normalized.

    Parameters
    ----------
    pad_sample
        The pad sample to return the faces.
    cropper : collections.abc.Callable
        A face image cropper that works with database's annotations.
    normalizer : collections.abc.Callable
        If not None, it should be a function that takes all the annotations of
        the whole video and yields normalized annotations frame by frame. It
        should yield same as ``annotations.items()``.

    Yields
    ------
    numpy.ndarray
        Face images

    Raises
    ------
    ValueError
        If the database returns None for annotations.
    """

    # read annotation
    annotations = pad_sample.annotations
    if annotations is None:
        raise ValueError("No annotations were returned.")

    if normalizer is not None:
        annotations = OrderedDict(normalizer(annotations))

    # normalize annotations and crop faces
    for frame_id, frame in enumerate(pad_sample.data):
        annot = annotations.get(str(frame_id), None)
        if annot is None:
            continue
        face = cropper(frame, annotations=annot)
        if face is not None:
            yield face


def scale_face(face, face_height, face_width=None):
    """Scales a face image to the given size.

    Parameters
    ----------
    face : numpy.ndarray
        The face image. It can be 2D or 3D in bob image format.
    face_height : int
        The height of the scaled face.
    face_width : :obj:`None`, optional
        The width of the scaled face. If None, face_height is used.

    Returns
    -------
    numpy.ndarray
        The scaled face.
    """
    face_width = face_height if face_width is None else face_width
    shape = list(face.shape)
    shape[-2:] = (face_height, face_width)
    # scaled_face = numpy.empty(shape, dtype="float64")
    scaled_face = scale(face, tuple(shape))
    return scaled_face


def blocks(data, block_size, block_overlap=(0, 0)):
    """Extracts patches of an image

    Parameters
    ----------
    data : numpy.ndarray
        The image in gray-scale, color, or color video format.
    block_size : (int, int)
        The size of patches
    block_overlap : (:obj:`int`, :obj:`int`), optional
        The size of overlap of patches

    Returns
    -------
    numpy.ndarray
        The patches.

    Raises
    ------
    ValueError
        If data dimension is not between 2 and 4 (inclusive).
    """

    data = numpy.asarray(data)
    # if a gray scale image:
    if data.ndim == 2:
        output = block(data, block_size, block_overlap, flat=True)
    # if a color image:
    elif data.ndim == 3:
        # out_shape = list(data.shape[0:1]) + list(
        #    block_output_shape(data[0], block_size, block_overlap, flat=True)
        # )

        # output = numpy.empty(out_shape, dtype=data.dtype)
        output = []
        for i, img2d in enumerate(data):
            output.append(block(img2d, block_size, block_overlap, flat=True))
        output = numpy.moveaxis(output, 0, 1)
    # if a color video:
    elif data.ndim == 4:
        output = [blocks(img3d, block_size, block_overlap) for img3d in data]
        output = numpy.concatenate(output, axis=0)
    else:
        raise ValueError("Unknown data dimension {}".format(data.ndim))
    return output


def block_generator(input, block_size, block_overlap=(0, 0)):
    """Performs a block decomposition of a 2D or 3D array/image

    It works exactly as :any:`bob.ip.base.block` except that it yields the blocks
    one by one instead of concatenating them. It also works with color images.

    Parameters
    ----------
    input : :any:`numpy.ndarray`
        A 2D array (Height, Width) or a color image (Bob format: Channels,
        Height, Width).
    block_size : (:obj:`int`, :obj:`int`)
        The size of the blocks in which the image is decomposed.
    block_overlap : (:obj:`int`, :obj:`int`), optional
        The overlap of the blocks.

    Yields
    ------
    array_like
        A block view of the image. Modifying the blocks will change the original
        image as well. This is different from :any:`bob.ip.base.block`.

    Raises
    ------
    ValueError
        If the block_overlap is not smaller than block_size.
        If the block_size is bigger than the image size.
    """
    block_h, block_w = block_size
    overlap_h, overlap_w = block_overlap
    img_h, img_w = input.shape[-2:]

    if overlap_h >= block_h or overlap_w >= block_w:
        raise ValueError(
            "block_overlap: {} must be smaller than block_size: {}.".format(
                block_overlap, block_size
            )
        )
    if img_h < block_h or img_w < block_w:
        raise ValueError(
            "block_size: {} must be smaller than the image size: {}.".format(
                block_size, input.shape[-2:]
            )
        )

    # Determine the number of block per row and column
    size_ov_h = block_h - overlap_h
    size_ov_w = block_w - overlap_w

    # Perform the block decomposition
    for h in range(0, img_h - block_h + 1, size_ov_h):
        for w in range(0, img_w - block_w + 1, size_ov_w):
            yield input[..., h : h + block_h, w : w + block_w]


def blocks_generator(data, block_size, block_overlap=(0, 0)):
    """Yields patches of an image

    Parameters
    ----------
    data : numpy.ndarray
        The image in gray-scale, color, or color video format.
    block_size : (int, int)
        The size of patches
    block_overlap : (:obj:`int`, :obj:`int`), optional
        The size of overlap of patches

    Yields
    ------
    numpy.ndarray
        The patches.

    Raises
    ------
    ValueError
        If data dimension is not between 2 and 4 (inclusive).
    """
    data = numpy.asarray(data)
    if 1 < data.ndim < 4:
        for patch in block_generator(data, block_size, block_overlap):
            yield patch
    # if a color video:
    elif data.ndim == 4:
        for frame in data:
            for patch in block_generator(frame, block_size, block_overlap):
                yield patch
    else:
        raise ValueError("Unknown data dimension {}".format(data.ndim))


def color_augmentation(image, channels=("rgb",)):
    """Converts an RGB image to different color channels.

    Parameters
    ----------
    image : numpy.ndarray
        The image in RGB Bob format.
    channels : :obj:`tuple`, optional
        List of channels to convert the image to. It can be any of ``rgb``,
        ``yuv``, ``hsv``.

    Returns
    -------
    numpy.ndarray
        The image that contains several channels:
        ``(3*len(channels), height, width)``.
    """
    final_image = []

    if "rgb" in channels:
        final_image.append(image)

    if "yuv" in channels:
        final_image.append(rgb_to_yuv(image))

    if "hsv" in channels:
        final_image.append(rgb_to_hsv(image))

    return numpy.concatenate(final_image, axis=0)


def random_sample(A, size):
    """Randomly selects ``size`` samples from the array ``A``"""
    return A[numpy.random.choice(A.shape[0], size, replace=False), ...]


def random_patches(image, block_size, n_random_patches=1):
    """Extracts N random patches of block_size from an image"""
    h, w = image.shape[-2:]
    bh, bw = block_size
    if h < block_size[0] or w < block_size[1]:
        raise ValueError("block_size must be smaller than image shape")
    hl = numpy.random.randint(0, h - bh, size=n_random_patches)
    wl = numpy.random.randint(0, w - bw, size=n_random_patches)
    for ch, cw in zip(hl, wl):
        yield image[..., ch : ch + bh, cw : cw + bw]


def extract_patches(
    image, block_size, block_overlap=(0, 0), n_random_patches=None
):
    """Yields either all patches from an image or N random patches."""
    if n_random_patches is None:
        return blocks_generator(image, block_size, block_overlap)
    else:
        return random_patches(
            image, block_size, n_random_patches=n_random_patches
        )


def the_giant_video_loader(
    pad_sample,
    region="whole",
    scaling_factor=None,
    cropper=None,
    normalizer=None,
    patches=False,
    block_size=(96, 96),
    block_overlap=(0, 0),
    random_patches_per_frame=None,
    augment=None,
    multiple_bonafide_patches=1,
    keep_pa_samples=None,
    keep_bf_samples=None,
):
    """Loads a video pad file frame by frame and optionally applies
    transformations.

    Parameters
    ----------
    pad_sample
        The pad sample
    region : str
        Either `whole` or `crop`. If whole, it will return the whole frame.
        Otherwise, you need to provide a cropper and a normalizer.
    scaling_factor : float
        If given, will scale images to this factor.
    cropper
        The cropper to use
    normalizer
        The normalizer to use
    patches : bool
        If true, will extract patches from images.
    block_size : tuple
        Size of the patches
    block_overlap : tuple
        Size of overlap of the patches
    random_patches_per_frame : int
        If not None, will only take this much patches per frame
    augment
        If given, frames will be transformed using this function.
    multiple_bonafide_patches : int
        Will use more random patches for bonafide samples
    keep_pa_samples : float
        If given, will drop some PA samples.
    keep_bf_samples : float
        If given, will drop some BF samples.

    Returns
    -------
    object
        A generator that yields the samples.

    Raises
    ------
    ValueError
        If region is not whole or crop.
    """
    if region == "whole":
        generator = iter(pad_sample.data)
    elif region == "crop":
        generator = yield_faces(
            pad_sample, cropper=cropper, normalizer=normalizer
        )
    else:
        raise ValueError("Invalid region value: `{}'".format(region))

    if scaling_factor is not None:
        generator = (scale(frame, scaling_factor) for frame in generator)
    if patches:
        if random_patches_per_frame is None:
            generator = (
                patch
                for frame in generator
                for patch in blocks_generator(frame, block_size, block_overlap)
            )
        else:
            if pad_sample.is_bonafide:
                random_patches_per_frame *= multiple_bonafide_patches
            generator = (
                patch
                for frame in generator
                for patch in random_sample(
                    blocks(frame, block_size, block_overlap),
                    random_patches_per_frame,
                )
            )

    if augment is not None:
        generator = (augment(frame) for frame in generator)

    if keep_pa_samples is not None and not pad_sample.is_bonafide:
        generator = (
            frame for frame in generator if random.random() < keep_pa_samples
        )

    if keep_bf_samples is not None and pad_sample.is_bonafide:
        generator = (
            frame for frame in generator if random.random() < keep_bf_samples
        )

    return generator
