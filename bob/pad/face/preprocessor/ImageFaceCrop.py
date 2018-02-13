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

#==============================================================================
# Main body:

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
    def __init__(self,
                 face_size,
                 rgb_output_flag = False):


        Preprocessor.__init__(self,
                              face_size = face_size,
                              rgb_output_flag = rgb_output_flag)

        self.face_size = face_size
        self.rgb_output_flag = rgb_output_flag


    #==========================================================================
    def normalize_image_size_in_grayscale(self, image, annotations, face_size):
        """
        This function crops the face in the input Gray-scale image given annotations
        defining the face bounding box. The size of the face is also normalized to the
        pre-defined dimensions.

        The algorithm is identical to the following paper:
        "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing"

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Gray-scale input image.

        ``annotations`` : :py:class:`dict`
            A dictionary containing annotations of the face bounding box.
            Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col)}``

        ``face_size`` : :py:class:`int`
            The size of the face after normalization.

        **Returns:**

        ``normbbx`` : 2D :py:class:`numpy.ndarray`
            An image of the cropped face of the size (self.face_size, self.face_size).
        """

        cutframe = image[annotations['topleft'][0]:annotations['bottomright'][0],
                 annotations['topleft'][1]:annotations['bottomright'][1]]

        tempbbx = np.ndarray((face_size, face_size), 'float64')
        normbbx = np.ndarray((face_size, face_size), 'uint8')
        bob.ip.base.scale(cutframe, tempbbx) # normalization
        tempbbx_ = tempbbx + 0.5
        tempbbx_ = np.floor(tempbbx_)
        normbbx = np.cast['uint8'](tempbbx_)

        return normbbx


    #==========================================================================
    def normalize_image_size(self, image, annotations, face_size, rgb_output_flag):
        """
        This function crops the face in the input image given annotations defining
        the face bounding box. The size of the face is also normalized to the
        pre-defined dimensions. For RGB inputs it is possible to return both
        color and gray-scale outputs. This option is controlled by ``rgb_output_flag``.

        The algorithm is identical to the following paper:
        "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing"

        **Parameters:**

        ``image`` : 2D or 3D :py:class:`numpy.ndarray`
            Input image (RGB or gray-scale).

        ``annotations`` : :py:class:`dict`
            A dictionary containing annotations of the face bounding box.
            Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col)}``

        ``face_size`` : :py:class:`int`
            The size of the face after normalization.

        ``rgb_output_flag`` : :py:class:`bool`
            Return RGB cropped face if ``True``, otherwise a gray-scale image is
            returned. Default: ``False``.

        **Returns:**

        ``face`` : 2D or 3D :py:class:`numpy.ndarray`
            An image of the cropped face of the size (self.face_size, self.face_size),
            rgb 3D or gray-scale 2D.
        """

        if len(image.shape) == 3:

            if not(rgb_output_flag):

                image = bob.ip.color.rgb_to_gray(image)

        if len(image.shape) == 2:

            image = [image] # make gray-scale image an iterable

        result = []

        for image_channel in image: # for all color channels in the input image

            cropped_face = self.normalize_image_size_in_grayscale(image_channel, annotations, face_size)

            result.append(cropped_face)

        face = np.stack(result, axis=0)

        face = np.squeeze(face) # squeeze 1-st dimension for gray-scale images

        return face


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

        norm_face_image = self.normalize_image_size(image, annotations, self.face_size, self.rgb_output_flag)

        return norm_face_image


