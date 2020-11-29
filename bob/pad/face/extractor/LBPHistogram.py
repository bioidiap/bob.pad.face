from bob.ip.base import LBP, histogram
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class LBPHistogram(TransformerMixin, BaseEstimator):
    """Calculates a normalized LBP histogram over an image.
    These features are implemented based on [CAM12]_.

    Parameters
    ----------
    lbp_type : str
        The type of the LBP operator (regular, uniform or riu2)
    elbp_type : str
        Which type of LBP codes should be computed; possible values: ('regular',
        'transitional', 'direction-coded'). For the old 'modified' method,
        specify `elbp_type` as 'regular` and `to_average` as True.
    to_average : bool
        Compare the neighbors to the average of the pixels instead of the central pixel?
    radius : float
        The radius of the circle on which the points are taken (for circular
        LBP)
    neighbors : int
        The number of points around the central point on which LBP is
        computed (4, 8, 16)
    circular : bool
        True if circular LBP is needed, False otherwise
    n_hor : int
        Number of blocks horizontally for spatially-enhanced LBP/MCT
        histograms. Default: 1
    n_vert
        Number of blocks vertically for spatially-enhanced LBP/MCT
        histograms. Default: 1

    Attributes
    ----------
    dtype : numpy.dtype
        If a ``dtype`` is specified in the contructor, it is assured that the
        resulting features have that dtype.
    lbp : LBP
        The LPB extractor object.
    """

    def __init__(
        self,
        lbp_type="uniform",
        elbp_type="regular",
        to_average=False,
        radius=1,
        neighbors=8,
        circular=False,
        dtype=None,
        n_hor=1,
        n_vert=1,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.lbp_type = lbp_type
        self.elbp_type = elbp_type
        self.to_average = to_average
        self.radius = radius
        self.neighbors = neighbors
        self.circular = circular
        self.dtype = dtype
        self.n_hor = n_hor
        self.n_vert = n_vert

        self.fit()

    def fit(self, X=None, y=None):

        self.lbp_ = LBP(
            neighbors=self.neighbors,
            radius=self.radius,
            circular=self.circular,
            to_average=self.to_average,
            uniform=self.lbp_type in ("uniform", "riu2"),
            rotation_invariant=self.lbp_type == "riu2",
            elbp_type=self.elbp_type,
        )
        return self

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("lbp_")
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fit()

    def comp_block_histogram(self, data):
        """
        Extracts LBP/MCT histograms from a gray-scale image/block.

        Takes data of arbitrary dimensions and linearizes it into a 1D vector;
        Then, calculates the histogram.
        enforcing the data type, if desired.

        Parameters
        ----------
        data : numpy.ndarray
            The preprocessed data to be transformed into one vector.

        Returns
        -------
        1D :py:class:`numpy.ndarray`
            The extracted feature vector, of the desired ``dtype`` (if
            specified)
        """
        assert isinstance(data, np.ndarray)

        # allocating the image with lbp codes
        lbpimage = np.ndarray(self.lbp_.lbp_shape(data), "uint16")
        self.lbp_(data, lbpimage)  # calculating the lbp image
        hist = histogram(lbpimage, (0, self.lbp_.max_label - 1), self.lbp_.max_label)
        hist = hist / np.sum(hist)  # histogram normalization
        if self.dtype is not None:
            hist = hist.astype(self.dtype)
        return hist

    def transform_one_image(self, data):
        """
        Extracts spatially-enhanced LBP/MCT histograms from a gray-scale image.

        Parameters
        ----------
        data : numpy.ndarray
            The preprocessed data to be transformed into one vector.

        Returns
        -------
        1D :py:class:`numpy.ndarray`
            The extracted feature vector, of the desired ``dtype`` (if
            specified)

        """

        # Make sure the data can be split into equal blocks:
        row_max = int(data.shape[0] / self.n_vert) * self.n_vert
        col_max = int(data.shape[1] / self.n_hor) * self.n_hor
        data = data[:row_max, :col_max]

        blocks = [
            sub_block
            for block in np.hsplit(data, self.n_hor)
            for sub_block in np.vsplit(block, self.n_vert)
        ]

        hists = [self.comp_block_histogram(block) for block in blocks]

        hist = np.hstack(hists)

        hist = hist / len(blocks)  # histogram normalization

        return hist

    def transform(self, images):
        return [self.transform_one_image(img) for img in images]

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
