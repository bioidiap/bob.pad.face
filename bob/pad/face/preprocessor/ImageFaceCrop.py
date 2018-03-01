#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:11:16 2017

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.preprocessor import Preprocessor

import numpy as np

import bob.ip.color

import bob.ip.base


#==========================================================================
def normalize_image_size_in_grayscale(image, annotations, face_size, use_face_alignment):
    """
    This function crops the face in the input Gray-scale image given annotations
    defining the face bounding box, and eye positions.
    The size of the face is also normalized to the pre-defined dimensions.

    Two normalization options are available, which are controlled by
    ``use_face_alignment`` flag, see below.

    **Parameters:**

    ``image`` : 2D :py:class:`numpy.ndarray`
        Gray-scale input image.

    ``annotations`` : :py:class:`dict`
        A dictionary containing annotations of the face bounding box,
        eye locations and facial landmarks.
        Dictionary must be as follows: ``{'topleft': (row, col), 'bottomright': (row, col),
        'left_eye': (row, col), 'right_eye': (row, col)}``.

    ``face_size`` : :py:class:`int`
        The size of the face after normalization.

    ``use_face_alignment`` : :py:class:`bool`
        If ``False``, the re-sizing from this publication is used:
        "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing"
        If ``True`` the facial image is both re-sized and aligned using
        positions of the eyes, which are given in the annotations.

    **Returns:**

    ``normbbx`` : 2D :py:class:`numpy.ndarray`
        An image of the cropped face of the size (face_size, face_size).
    """

    if use_face_alignment:

        face_eyes_norm = bob.ip.base.FaceEyesNorm(
            eyes_distance=((face_size + 1) / 2.),
            crop_size=(face_size, face_size),
            eyes_center=(face_size / 4., (face_size - 0.5) / 2.))

        right_eye, left_eye = annotations['right_eye'], annotations['left_eye']

        normalized_image = face_eyes_norm( image, right_eye = right_eye, left_eye = left_eye )

        normbbx=normalized_image.astype('uint8')

    else:

        cutframe = image[annotations['topleft'][0]:annotations['bottomright'][
            0], annotations['topleft'][1]:annotations['bottomright'][1]]

        tempbbx = np.ndarray((face_size, face_size), 'float64')
        normbbx = np.ndarray((face_size, face_size), 'uint8')
        bob.ip.base.scale(cutframe, tempbbx)  # normalization
        tempbbx_ = tempbbx + 0.5
        tempbbx_ = np.floor(tempbbx_)
        normbbx = np.cast['uint8'](tempbbx_)

    return normbbx


#==========================================================================
def normalize_image_size(image, annotations, face_size,
                         rgb_output_flag, use_face_alignment):
    """
    This function crops the face in the input image given annotations defining
    the face bounding box. The size of the face is also normalized to the
    pre-defined dimensions. For RGB inputs it is possible to return both
    color and gray-scale outputs. This option is controlled by ``rgb_output_flag``.

    Two normalization options are available, which are controlled by
    ``use_face_alignment`` flag, see below.

    **Parameters:**

    ``image`` : 2D or 3D :py:class:`numpy.ndarray`
        Input image (RGB or gray-scale).

    ``annotations`` : :py:class:`dict`
        A dictionary containing annotations of the face bounding box,
        eye locations and facial landmarks.
        Dictionary must be as follows: ``{'topleft': (row, col), 'bottomright': (row, col),
        'left_eye': (row, col), 'right_eye': (row, col)}``.

    ``face_size`` : :py:class:`int`
        The size of the face after normalization.

    ``rgb_output_flag`` : :py:class:`bool`
        Return RGB cropped face if ``True``, otherwise a gray-scale image is
        returned. Default: ``False``.

    ``use_face_alignment`` : :py:class:`bool`
        If ``False``, the facial image re-sizing from this publication is used:
        "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing"
        If ``True`` the facial image is both re-sized, and aligned, using
        positions of the eyes, which are given in the annotations.

    **Returns:**

    ``face`` : 2D or 3D :py:class:`numpy.ndarray`
        An image of the cropped face of the size (face_size, face_size),
        RGB 3D or gray-scale 2D.
    """

    if len(image.shape) == 3:

        if not (rgb_output_flag):

            image = bob.ip.color.rgb_to_gray(image)

    if len(image.shape) == 2:

        image = [image]  # make gray-scale image an iterable

    result = []

    for image_channel in image:  # for all color channels in the input image

        cropped_face = normalize_image_size_in_grayscale(
            image_channel, annotations, face_size, use_face_alignment)

        result.append(cropped_face)

    face = np.stack(result, axis=0)

    face = np.squeeze(face)  # squeeze 1-st dimension for gray-scale images

    return face








class ImageFaceCrop(Preprocessor):
    """
    This class crops the face in the input image given annotations defining
    the face bounding box. The size of the face is also normalized to the
    pre-defined dimensions. For RGB inputs it is possible to return both
    color and gray-scale outputs. This option is controlled by ``rgb_output_flag``.

    The algorithm is identical to the following paper:
    "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing"

    **Parameters:**

    ``face_size`` : :py:class:`int`
        The size of the face after normalization.

    ``rgb_output_flag`` : :py:class:`bool`
        Return RGB cropped face if ``True``, otherwise a gray-scale image is
        returned. Default: ``False``.
    """

    #==========================================================================
    def __init__(self, face_size, rgb_output_flag=False):

        Preprocessor.__init__(
            self, face_size=face_size, rgb_output_flag=rgb_output_flag)

        self.face_size = face_size
        self.rgb_output_flag = rgb_output_flag


    #==========================================================================
    def __call__(self, image, annotations):
        """
        Call the ``normalize_image_size()`` method of this class.

        **Parameters:**

        ``image`` : 2D or 3D :py:class:`numpy.ndarray`
            Input image (RGB or gray-scale).

        ``annotations`` : :py:class:`dict`
            A dictionary containing annotations of the face bounding box.
            Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col)}``

        **Returns:**

        ``norm_face_image`` : 2D or 3D :py:class:`numpy.ndarray`
            An image of the cropped face of the size (self.face_size, self.face_size),
            rgb 3D or gray-scale 2D.
        """

        norm_face_image = self.normalize_image_size(
            image, annotations, self.face_size, self.rgb_output_flag,
            self.use_face_alignment)

        return norm_face_image










    #==========================================================================
    def __call__(self, frames, annotations):
        """
        Crop the face in the input video frames given annotations for each frame.

        **Parameters:**

        ``frames`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box in frame N.

        **Returns:**

        ``preprocessed_video`` : FrameContainer
            Cropped faces stored in the FrameContainer.
        """







        if self.detect_faces_flag:

            if self.max_image_size: # max_image_size = 1920

                if np.max(frames[0][1].shape) > self.max_image_size:

                    return bob.bio.video.FrameContainer()

            try:

                annotations = detect_face_landmarks_in_video(frames,
                                                    self.face_detection_method) #TODO: new dicts
            except:

                return bob.bio.video.FrameContainer()

        if len(frames) != len(annotations):  # if some annotations are missing

            ## Select only annotated frames:
            frames, annotations = self.select_annotated_frames(
                frames, annotations)

        preprocessed_video = self.video_preprocessor(
            frames=frames, annotations=annotations)

        if self.check_face_size_flag:

            preprocessed_video = self.check_face_size(
                preprocessed_video, annotations, self.min_face_size)

        return preprocessed_video



