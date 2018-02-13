#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:47:47 2017

@author: Olegs Nikisins
"""

# ==============================================================================
# Import what is needed here:

from bob.pad.base.algorithm import Algorithm

from bob.bio.video.utils import FrameContainer

import numpy as np

import bob.io.base

from sklearn import mixture

# ==============================================================================
# Main body :


class VideoGmmPadAlgorithm(Algorithm):
    """
    This class is designed to train a GMM based PAD system. The GMM is trained
    using data of one class (real class) only. The procedure is the following:

    1. First, the training data is mean-std normalized using mean and std of the
       real class only.

    2. Second, the GMM with ``n_components`` Gaussians is trained using samples
       of the real class.

    3. The input features are next classified using pre-trained GMM machine.

    **Parameters:**

    ``n_components`` : :py:class:`int`
        Number of Gaussians in the GMM. Default: 1 .

    ``random_state`` : :py:class:`int`
        A seed for the random number generator used in the initialization of
        the GMM. Default: 7 .

    ``frame_level_scores_flag`` : :py:class:`bool`
        Return scores for each frame individually if True. Otherwise, return a
        single score per video. Default: False.
    """

    def __init__(self,
                 n_components=1,
                 random_state=3,
                 frame_level_scores_flag=False):

        Algorithm.__init__(
            self,
            n_components=n_components,
            random_state=random_state,
            frame_level_scores_flag=frame_level_scores_flag,
            performs_projection=True,
            requires_projector_training=True)

        self.n_components = n_components

        self.random_state = random_state

        self.frame_level_scores_flag = frame_level_scores_flag

        self.machine = None  # this argument will be updated with pretrained GMM machine

        self.features_mean = None  # this argument will be updated with features mean

        self.features_std = None  # this argument will be updated with features std

        # names of the arguments of the pretrained GMM machine to be saved/loaded to/from HDF5 file:
        self.gmm_param_keys = [
            "covariance_type", "covariances_", "lower_bound_", "means_",
            "n_components", "weights_", "converged_", "precisions_",
            "precisions_cholesky_"
        ]

    # ==========================================================================
    def convert_frame_cont_to_array(self, frame_container):
        """
        This function converts a single Frame Container into an array of features.
        The rows are samples, the columns are features.

        **Parameters:**

        ``frame_container`` : object
            A Frame Container conteining the features of an individual,
            see ``bob.bio.video.utils.FrameContainer``.

        **Returns:**

        ``features_array`` : 2D :py:class:`numpy.ndarray`
            An array containing features for all frames.
            The rows are samples, the columns are features.
        """

        feature_vectors = []

        frame_dictionary = {}

        for frame in frame_container:
            frame_dictionary[frame[0]] = frame[1]

        for idx, _ in enumerate(frame_container):
            # Frames are stored in a mixed order, therefore we get them using incrementing frame index:
            feature_vectors.append(frame_dictionary[str(idx)])

        features_array = np.vstack(feature_vectors)

        return features_array

    # ==========================================================================
    def convert_and_prepare_features(self, features):
        """
        This function converts a list or a frame container of features into a 2D array of features.
        If the input is a list of frame containers, features from different frame containers (individuals)
        are concatenated into the same list. This list is then converted to an array. The rows are samples,
        the columns are features.

        **Parameters:**

        ``features`` : [2D :py:class:`numpy.ndarray`] or [FrameContainer]
            A list or 2D feature arrays or a list of Frame Containers, see ``bob.bio.video.utils.FrameContainer``.
            Each frame Container contains feature vectors for the particular individual/person.

        **Returns:**

        ``features_array`` : 2D :py:class:`numpy.ndarray`
            An array containing features for all samples and frames.
        """

        if isinstance(
                features[0],
                FrameContainer):  # if FrameContainer convert to 2D numpy array
            return self.convert_list_of_frame_cont_to_array(features)
        else:
            return np.vstack(features)

    # ==========================================================================
    def convert_list_of_frame_cont_to_array(self, frame_containers):
        """
        This function converts a list of Frame containers into an array of features.
        Features from different frame containers (individuals) are concatenated into the
        same list. This list is then converted to an array. The rows are samples,
        the columns are features.

        **Parameters:**

        ``frame_containers`` : [FrameContainer]
            A list of Frame Containers, , see ``bob.bio.video.utils.FrameContainer``.
            Each frame Container contains feature vectors for the particular individual/person.

        **Returns:**

        ``features_array`` : 2D :py:class:`numpy.ndarray`
            An array containing features for all frames of all individuals.
        """

        feature_vectors = []

        for frame_container in frame_containers:
            video_features_array = self.convert_frame_cont_to_array(
                frame_container)

            feature_vectors.append(video_features_array)

        features_array = np.vstack(feature_vectors)

        return features_array

    # ==========================================================================
    def mean_std_normalize(self,
                           features,
                           features_mean=None,
                           features_std=None):
        """
        The features in the input 2D array are mean-std normalized.
        The rows are samples, the columns are features. If ``features_mean``
        and ``features_std`` are provided, then these vectors will be used for
        normalization. Otherwise, the mean and std of the features is
        computed on the fly.

        **Parameters:**

        ``features`` : 2D :py:class:`numpy.ndarray`
            Array of features to be normalized.

        ``features_mean`` : 1D :py:class:`numpy.ndarray`
            Mean of the features. Default: None.

        ``features_std`` : 2D :py:class:`numpy.ndarray`
            Standart deviation of the features. Default: None.

        **Returns:**

        ``features_norm`` : 2D :py:class:`numpy.ndarray`
            Normalized array of features.

        ``features_mean`` : 1D :py:class:`numpy.ndarray`
            Mean of the features.

        ``features_std`` : 1D :py:class:`numpy.ndarray`
            Standart deviation of the features.
        """

        features = np.copy(features)

        # Compute mean and std if not given:
        if features_mean is None:
            features_mean = np.mean(features, axis=0)

            features_std = np.std(features, axis=0)

        row_norm_list = []

        for row in features:  # row is a sample

            row_norm = (row - features_mean) / features_std

            row_norm_list.append(row_norm)

        features_norm = np.vstack(row_norm_list)

        return features_norm, features_mean, features_std

    # ==========================================================================
    def train_gmm(self, real, n_components, random_state):
        """
        Train GMM classifier given real class. Prior to the training the data is
        mean-std normalized.

        **Parameters:**

        ``real`` : 2D :py:class:`numpy.ndarray`
            Training features for the real class.

        ``n_components`` : :py:class:`int`
            Number of Gaussians in the GMM. Default: 1 .

        ``random_state`` : :py:class:`int`
            A seed for the random number generator used in the initialization of
            the GMM. Default: 7 .

        **Returns:**

        ``machine`` : object
            A trained GMM machine.

        ``features_mean`` : 1D :py:class:`numpy.ndarray`
            Mean of the features.

        ``features_std`` : 1D :py:class:`numpy.ndarray`
            Standart deviation of the features.
        """

        features_norm, features_mean, features_std = self.mean_std_normalize(
            real)
        # real is now mean-std normalized

        machine = mixture.GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            covariance_type='full')

        machine.fit(features_norm)

        return machine, features_mean, features_std

    # ==========================================================================
    def save_gmm_machine_and_mean_std(self, projector_file, machine,
                                      features_mean, features_std):
        """
        Saves the GMM machine, features mean and std to the hdf5 file.
        The absolute name of the file is specified in ``projector_file`` string.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            Absolute name of the file to save the data to, as returned by
            ``bob.pad.base`` framework.

        ``machine`` : object
            The GMM machine to be saved. As returned by sklearn.linear_model
            module.

        ``features_mean`` : 1D :py:class:`numpy.ndarray`
            Mean of the features.

        ``features_std`` : 1D :py:class:`numpy.ndarray`
            Standart deviation of the features.
        """

        f = bob.io.base.HDF5File(projector_file,
                                 'w')  # open hdf5 file to save to

        for key in self.gmm_param_keys:
            data = getattr(machine, key)

            f.set(key, data)

        f.set("features_mean", features_mean)

        f.set("features_std", features_std)

        del f

    # ==========================================================================
    def train_projector(self, training_features, projector_file):
        """
        Train GMM for feature projection and save it to file.
        The ``requires_projector_training = True`` flag must be set to True
        to enable this function.

        **Parameters:**

        ``training_features`` : [[FrameContainer], [FrameContainer]]
            A list containing two elements: [0] - a list of Frame Containers with
            feature vectors for the real class; [1] - a list of Frame Containers with
            feature vectors for the attack class.

        ``projector_file`` : :py:class:`str`
            The file to save the trained projector to, as returned by the
            ``bob.pad.base`` framework.
        """

        # training_features[0] - training features for the REAL class.
        real = self.convert_and_prepare_features(
            training_features[0])  # output is array

        # training_features[1] - training features for the ATTACK class.
        #        attack = self.convert_and_prepare_features(training_features[1]) # output is array

        # Train the GMM machine and get normalizers:
        machine, features_mean, features_std = self.train_gmm(
            real=real,
            n_components=self.n_components,
            random_state=self.random_state)

        # Save the GNN machine and normalizers:
        self.save_gmm_machine_and_mean_std(projector_file, machine,
                                           features_mean, features_std)

    # ==========================================================================
    def load_gmm_machine_and_mean_std(self, projector_file):
        """
        Loads the machine, features mean and std from the hdf5 file.
        The absolute name of the file is specified in ``projector_file`` string.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            Absolute name of the file to load the trained projector from, as
            returned by ``bob.pad.base`` framework.

        **Returns:**

        ``machine`` : object
            The loaded GMM machine. As returned by sklearn.mixture module.

        ``features_mean`` : 1D :py:class:`numpy.ndarray`
            Mean of the features.

        ``features_std`` : 1D :py:class:`numpy.ndarray`
            Standart deviation of the features.
        """

        f = bob.io.base.HDF5File(projector_file,
                                 'r')  # file to read the machine from

        # initialize the machine:
        machine = mixture.GaussianMixture()

        # set the params of the machine:
        for key in self.gmm_param_keys:
            data = f.read(key)

            setattr(machine, key, data)

        features_mean = f.read("features_mean")

        features_std = f.read("features_std")

        del f

        return machine, features_mean, features_std

    # ==========================================================================
    def load_projector(self, projector_file):
        """
        Loads the machine, features mean and std from the hdf5 file.
        The absolute name of the file is specified in ``projector_file`` string.

        This function sets the arguments ``self.machine``, ``self.features_mean``
        and ``self.features_std`` of this class with loaded machines.

        The function must be capable of reading the data saved with the
        :py:meth:`train_projector` method of this class.

        Please register `performs_projection = True` in the constructor to
        enable this function.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            The file to read the projector from, as returned by the
            ``bob.pad.base`` framework. In this class the names of the files to
            read the projectors from are modified, see ``load_machine`` and
            ``load_cascade_of_machines`` methods of this class for more details.
        """

        machine, features_mean, features_std = self.load_gmm_machine_and_mean_std(
            projector_file)

        self.machine = machine

        self.features_mean = features_mean

        self.features_std = features_std

    # ==========================================================================
    def project(self, feature):
        """
        This function computes a vector of scores for each sample in the input
        array of features. The following steps are applied:

        1. First, the input data is mean-std normalized using mean and std of the
           real class only.

        2. The input features are next classified using pre-trained GMM machine.

        Set ``performs_projection = True`` in the constructor to enable this function.
        It is assured that the :py:meth:`load_projector` was **called before** the
        ``project`` function is executed.

        **Parameters:**

        ``feature`` : FrameContainer or 2D :py:class:`numpy.ndarray`
            Two types of inputs are accepted.
            A Frame Container conteining the features of an individual,
            see ``bob.bio.video.utils.FrameContainer``.
            Or a 2D feature array of the size (N_samples x N_features).

        **Returns:**

        ``scores`` : 1D :py:class:`numpy.ndarray`
            Vector of scores. Scores for the real class are expected to be
            higher, than the scores of the negative / attack class.
            In this case scores are the weighted log probabilities.
        """

        # 1. Convert input array to numpy array if necessary.
        if isinstance(
                feature,
                FrameContainer):  # if FrameContainer convert to 2D numpy array

            features_array = self.convert_frame_cont_to_array(feature)

        else:

            features_array = feature

        features_array_norm, _, _ = self.mean_std_normalize(
            features_array, self.features_mean, self.features_std)

        scores = self.machine.score_samples(features_array_norm)

        return scores

    # ==========================================================================
    def score(self, toscore):
        """
        Returns a probability of a sample being a real class.

        **Parameters:**

        ``toscore`` : 1D :py:class:`numpy.ndarray`
            Vector with scores for each frame/sample defining the probability
            of the frame being a sample of the real class.

        **Returns:**

        ``score`` : [:py:class:`float`]
            If ``frame_level_scores_flag = False`` a single score is returned.
            One score per video. This score is placed into a list, because
            the ``score`` must be an iterable.
            Score is a probability of a sample being a real class.
            If ``frame_level_scores_flag = True`` a list of scores is returned.
            One score per frame/sample.
        """

        if self.frame_level_scores_flag:

            score = list(toscore)

        else:

            score = [np.mean(toscore)]  # compute a single score per video

        return score
