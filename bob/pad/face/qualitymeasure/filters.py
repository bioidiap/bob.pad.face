# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
import scipy
import numpy as np


def sobel(image):
    """
    Implements the sobel filter like bob.ip.base.sobel

    Performs a Sobel filtering of the input image
    This function will perform a Sobel filtering woth both the vertical and the horizontal filter.
    A Sobel filter is an edge detector, which will detect either horizontal or vertical edges.
    The two filter are given as:

    .. math:: S_y =  \\left\\lgroup\\begin{array}{ccc} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{array}\\right\\rgroup \\qquad S_x = \\left\\lgroup\\begin{array}{ccc} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{array}\\right\\rgroup\n\n
    If given, the dst array should have the expected type (numpy.float64) and two layers of the same size as the input image.
    Finally, the result of the vertical filter will be put into the first layer of ``dst[0]``, while the result of the horizontal filter will be written to ``dst[1]``.
    """

    mask_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    mask_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    return np.array(
        [
            scipy.ndimage.convolve(image, weights=mask_v),
            scipy.ndimage.convolve(image, weights=mask_h),
        ]
    )
