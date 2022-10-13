import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


def _get_cropping_size(image_size, patch_size):
    # How many pixels missing to cover the whole image
    r = image_size % patch_size
    # Spit gap into two evenly
    before = r // 2
    after = image_size - (r - before)
    return before, after


def _extract_patches(image, patch_size):
    # https://stackoverflow.com/a/16858283
    h, w = image.shape
    nrows, ncols = patch_size
    if h % nrows != 0 or w % ncols != 0:
        w_left, w_right = _get_cropping_size(w, ncols)
        h_top, h_bottom = _get_cropping_size(h, nrows)
        # Perform center crop
        image = image[h_top:h_bottom, w_left:w_right]
    return (
        image.reshape(h // nrows, nrows, -1, ncols)
        .swapaxes(1, 2)
        .reshape(-1, nrows, ncols)
    )


class SpatialHistogram(TransformerMixin, BaseEstimator):
    """
    Split images into a grid of patches, compute histogram on each one of them
    and concatenate them to obtain the final descriptor.
    """

    def __init__(self, grid_size=(4, 4), range=(0, 256), nbins=256):
        """
        Constructor
        :param grid_size: Tuple `(grid_y, grid_x)` indicating the number of
            patches to extract in each directions
        :param range: Tuple `(h_min, h_max)` indicating the histogram range.
            cf numpy.histogram
        :param nbins: Number of bins in the histogram, cf numpy.histogram
        """
        self.grid_size = grid_size
        self.range = range
        self.nbins = nbins

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = check_array(X, allow_nd=True)  # X.shape == (N, H, W)
        histo = []
        for sample in X:
            h = self._spatial_histogram(sample)  # [grid_x * grid_y * nbins]
            histo.append(h)
        return np.asarray(histo)

    def _spatial_histogram(self, image):
        """Compute spatial histogram for a given images"""
        patch_size = [s // g for s, g in zip(image.shape, self.grid_size)]
        patches = _extract_patches(image=image, patch_size=patch_size)
        hist = []
        for patch in patches:
            h, _ = np.histogram(
                patch, bins=self.nbins, range=self.range, density=True
            )
            hist.append(h)
        return np.asarray(hist).reshape(-1)

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
