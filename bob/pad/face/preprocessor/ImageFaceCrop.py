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
    pre-defined dimensions. If input image is RGB it is first converted to the
    gray-scale format.
    The algorithm is identical to the following paper:
    "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing"

    **Parameters:**

    ``face_size`` : :py:class:`int`
        The size of the face after normalization.
    """
    #==========================================================================
    def __init__(self, face_size):


        Preprocessor.__init__(self,
                              face_size = face_size)

        self.face_size = face_size


    #==========================================================================
    def normalize_image_size(self, image, annotations, face_size):
        """
        This function crops the face in the input image given annotations defining
        the face bounding box. The size of the face is also normalized to the
        pre-defined dimensions. If input image is RGB it is first converted to the
        gray-scale format.
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

        **Returns:**

        ``normbbx`` : 2D :py:class:`numpy.ndarray`
            An image of the cropped face of the size (face_size, face_size).
        """

        if len(image.shape) == 3:

            image = bob.ip.color.rgb_to_gray(image)

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

        ``norm_face_image`` : 2D :py:class:`numpy.ndarray`
            An image of the cropped face of the size (self.face_size, self.face_size).
        """

        norm_face_image = self.normalize_image_size(image, annotations, self.face_size)

        return norm_face_image


