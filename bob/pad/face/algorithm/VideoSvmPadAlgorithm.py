#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:43:09 2017

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

from bob.pad.base.algorithm import Algorithm

import itertools as it

import numpy as np

import bob.learn.libsvm

import bob.io.base

import os

#==============================================================================
# Main body :

class VideoSvmPadAlgorithm(Algorithm):
    """
    This class is designed to train SVM given Frame Containers with features
    of real and attack classes. The trained SVM is then used to classify the
    testing data as either real or attack. The SVM is trained in two stages.
    First, the best parameters for SVM are estimated using train and
    cross-validation subsets. The size of the subsets used in hyper-parameter
    tuning is defined by ``n_samples`` parameter of this class. Once best
    parameters are determined, the SVM machine is trained using complete training
    set.

    **Parameters:**

    ``machine_type`` : :py:class:`str`
        A type of the SVM machine. Please check ``bob.learn.libsvm`` for
        more details. Default: 'C_SVC'.

    ``kernel_type`` : :py:class:`str`
        A type of kerenel for the SVM machine. Please check ``bob.learn.libsvm``
        for more details. Default: 'RBF'.

    ``n_samples`` : :py:class:`int`
        Number of uniformly selected feature vectors per class defining the
        sizes of sub-sets used in the hyper-parameter grid search.

    ``trainer_grid_search_params`` : :py:class:`dict`
        Dictionary containing the hyper-parameters of the SVM to be tested
        in the grid-search.
        Default: {'cost': [2**p for p in range(-5, 16, 2)], 'gamma': [2**p for p in range(-15, 4, 2)]}.

    ``mean_std_norm_flag`` : :py:class:`bool`
        Perform mean-std normalization of data if set to True. Default: False.

    ``frame_level_scores_flag`` : :py:class:`bool`
        Return scores for each frame individually if True. Otherwise, return a
        single score per video. Default: False.

    ``save_debug_data_flag`` : :py:class:`bool`
        Save the data, which might be usefull for debugging if ``True``.
        Default: ``True``.

    ``reduced_train_data_flag`` : :py:class:`bool`
        Reduce the amount of final training samples if set to ``True``.
        Default: ``False``.

    ``n_train_samples`` : :py:class:`int`
        Number of uniformly selected feature vectors per class defining the
        sizes of sub-sets used in the final traing of the SVM.
        Default: 50000.
    """

    def __init__(self,
                 machine_type = 'C_SVC',
                 kernel_type = 'RBF',
                 n_samples = 10000,
                 trainer_grid_search_params = { 'cost': [2**p for p in range(-5, 16, 2)], 'gamma': [2**p for p in range(-15, 4, 2)]},
                 mean_std_norm_flag = False,
                 frame_level_scores_flag = False,
                 save_debug_data_flag = True,
                 reduced_train_data_flag = False,
                 n_train_samples = 50000):


        Algorithm.__init__(self,
                           machine_type = machine_type,
                           kernel_type = kernel_type,
                           n_samples = n_samples,
                           trainer_grid_search_params = trainer_grid_search_params,
                           mean_std_norm_flag = mean_std_norm_flag,
                           frame_level_scores_flag = frame_level_scores_flag,
                           save_debug_data_flag = save_debug_data_flag,
                           reduced_train_data_flag = reduced_train_data_flag,
                           n_train_samples = n_train_samples,
                           performs_projection=True,
                           requires_projector_training=True)

        self.machine_type = machine_type
        self.kernel_type = kernel_type
        self.n_samples = n_samples
        self.trainer_grid_search_params = trainer_grid_search_params
        self.mean_std_norm_flag = mean_std_norm_flag
        self.frame_level_scores_flag = frame_level_scores_flag
        self.save_debug_data_flag = save_debug_data_flag
        self.reduced_train_data_flag = reduced_train_data_flag
        self.n_train_samples = n_train_samples
        self.machine = None


    #==========================================================================
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


    #==========================================================================
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

            video_features_array = self.convert_frame_cont_to_array(frame_container)

            feature_vectors.append( video_features_array )

        features_array = np.vstack(feature_vectors)

        return features_array


    #==========================================================================
    def combinations(self, input_dict):
        """
        Obtain all possible key-value combinations in the input dictionary
        containing list values.

        **Parameters:**

        ``input_dict`` : :py:class:`dict`
            Input dictionary with list values.

        **Returns:**

        ``combinations`` : [:py:class:`dict`]
            A list of dictionaries containing the combinations.
        """

        varNames = sorted(input_dict)

        combinations = [ dict( zip( varNames, prod ) ) for prod in it.product( *( input_dict[ varName ] for varName in varNames ) ) ]

        return combinations


    #==========================================================================
    def select_uniform_data_subset(self, features, n_samples):
        """
        Uniformly select N samples/feature vectors from the input array of samples.
        The rows in the input array are samples. The columns are features.

        **Parameters:**

        ``features`` : 2D :py:class:`numpy.ndarray`
            Input array with feature vectors. The rows are samples, columns are features.

        ``n_samples`` : :py:class:`int`
            The number of samples to be selected uniformly from the input array of features.

        **Returns:**

        ``features_subset`` : 2D :py:class:`numpy.ndarray`
            Selected subset of features.
        """

        if features.shape[0] <= n_samples:

            features_subset = features

        else:

            uniform_step = np.int(features.shape[0]/n_samples)

            features_subset = features[0 : np.int(uniform_step*n_samples) : uniform_step, :]

        return features_subset


    #==========================================================================
    def select_quasi_uniform_data_subset(self, features, n_samples):
        """
        Select quasi uniformly N samples/feature vectors from the input array of samples.
        The rows in the input array are samples. The columns are features.
        Use this function if n_samples is close to the number of samples.

        **Parameters:**

        ``features`` : 2D :py:class:`numpy.ndarray`
            Input array with feature vectors. The rows are samples, columns are features.

        ``n_samples`` : :py:class:`int`
            The number of samples to be selected uniformly from the input array of features.

        **Returns:**

        ``features_subset`` : 2D :py:class:`numpy.ndarray`
            Selected subset of features.
        """

        if features.shape[0] <= n_samples:

            features_subset = features

        else:

            uniform_step = (1.0 * features.shape[0]) / n_samples

            element_num_list = range(0,n_samples)

            idx = [np.int(uniform_step*item) for item in element_num_list]

            features_subset = features[idx, :]

        return features_subset


    #==========================================================================
    def split_data_to_train_cv(self, features):
        """
        This function is designed to split the input array of features into two
        subset namely train and cross-validation. These subsets can be used to tune the
        hyper-parameters of the SVM. The splitting is 50/50, the first half of the
        samples in the input are selected to be train set, and the second half of
        samples is cross-validation.

        **Parameters:**

        ``features`` : 2D :py:class:`numpy.ndarray`
            Input array with feature vectors. The rows are samples, columns are features.

        **Returns:**

        ``features_train`` : 2D :py:class:`numpy.ndarray`
            Selected subset of train features.

        ``features_cv`` : 2D :py:class:`numpy.ndarray`
            Selected subset of cross-validation features.
        """

        half_samples_num = np.int(features.shape[0]/2)

        features_train = features[ 0 : half_samples_num, : ]
        features_cv = features[ half_samples_num : 2 * half_samples_num + 1, : ]

        return features_train, features_cv


    #==========================================================================
    def prepare_data_for_hyper_param_grid_search(self, training_features, n_samples):
        """
        This function converts a list of all training features returned by ``read_features``
        method of the extractor to the subsampled train and cross-validation arrays for both
        real and attack classes.

        **Parameters:**

        ``training_features`` : [[FrameContainer], [FrameContainer]]
            A list containing two elements: [0] - a list of Frame Containers with
            feature vectors for the real class; [1] - a list of Frame Containers with
            feature vectors for the attack class.

        ``n_samples`` : :py:class:`int`
            Number of uniformly selected feature vectors per class.

        **Returns:**

        ``real_train`` : 2D :py:class:`numpy.ndarray`
            Selected subset of train features for the real class.
            The number of samples in this set is n_samples/2, which is defined
            by split_data_to_train_cv method of this class.

        ``real_cv`` : 2D :py:class:`numpy.ndarray`
            Selected subset of cross-validation features for the real class.
            The number of samples in this set is n_samples/2, which is defined
            by split_data_to_train_cv method of this class.

        ``attack_train`` : 2D :py:class:`numpy.ndarray`
            Selected subset of train features for the attack class.
            The number of samples in this set is n_samples/2, which is defined
            by split_data_to_train_cv method of this class.

        ``attack_cv`` : 2D :py:class:`numpy.ndarray`
            Selected subset of cross-validation features for the attack class.
            The number of samples in this set is n_samples/2, which is defined
            by split_data_to_train_cv method of this class.
        """

        # training_features[0] - training features for the REAL class.
        real = self.convert_list_of_frame_cont_to_array(training_features[0]) # output is array
        # training_features[1] - training features for the ATTACK class.
        attack = self.convert_list_of_frame_cont_to_array(training_features[1]) # output is array

        # uniformly select subsets of features:
        real_subset = self.select_uniform_data_subset(real, n_samples)
        attack_subset = self.select_uniform_data_subset(attack, n_samples)

        # split the data into train and cross-validation:
        real_train, real_cv = self.split_data_to_train_cv(real_subset)
        attack_train, attack_cv = self.split_data_to_train_cv(attack_subset)

        return real_train, real_cv, attack_train, attack_cv


    #==========================================================================
    def comp_prediction_precision(self, machine, real, attack):
        """
        This function computes the precision of the predictions as a ratio
        of correctly classified samples to the total number of samples.

        **Parameters:**

        ``machine`` : object
            A pre-trained SVM machine.

        ``real`` : 2D :py:class:`numpy.ndarray`
            Array of features representing the real class.

        ``attack`` : 2D :py:class:`numpy.ndarray`
            Array of features representing the attack class.

        **Returns:**

        ``precision`` : :py:class:`float`
            The precision of the predictions.
        """

        labels_real = machine.predict_class(real)

        labels_attack = machine.predict_class(attack)

        samples_num = len(labels_real) + len(labels_attack)

        precision = ( np.sum(labels_real == 1) + np.sum(labels_attack == -1) ).astype( np.float ) / samples_num

        return precision


    #==========================================================================
    def mean_std_normalize(self, features, features_mean= None, features_std = None):
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

        ``features_std`` : 2D :py:class:`numpy.ndarray`
            Standart deviation of the features.
        """

        features = np.copy(features)

        # Compute mean and std if not given:
        if features_mean is None:

            features_mean = np.mean(features, axis=0)

            features_std = np.std(features, axis=0)

        row_norm_list = []

        for row in features: # row is a sample

            row_norm = (row - features_mean) / features_std

            row_norm_list.append(row_norm)

        features_norm = np.vstack(row_norm_list)

        return features_norm, features_mean, features_std


    #==========================================================================
    def norm_train_cv_data(self, real_train, real_cv, attack_train, attack_cv):
        """
        Mean-std normalization of train and cross-validation data arrays.

        **Parameters:**

        ``real_train`` : 2D :py:class:`numpy.ndarray`
            Subset of train features for the real class.

        ``real_cv`` : 2D :py:class:`numpy.ndarray`
            Subset of cross-validation features for the real class.

        ``attack_train`` : 2D :py:class:`numpy.ndarray`
            Subset of train features for the attack class.

        ``attack_cv`` : 2D :py:class:`numpy.ndarray`
            Subset of cross-validation features for the attack class.

        **Returns:**

        ``real_train_norm`` : 2D :py:class:`numpy.ndarray`
            Normalized subset of train features for the real class.

        ``real_cv_norm`` : 2D :py:class:`numpy.ndarray`
            Normalized subset of cross-validation features for the real class.

        ``attack_train_norm`` : 2D :py:class:`numpy.ndarray`
            Normalized subset of train features for the attack class.

        ``attack_cv_norm`` : 2D :py:class:`numpy.ndarray`
            Normalized subset of cross-validation features for the attack class.
        """

        features_train = np.vstack([real_train, attack_train])

        features_train_norm, features_mean, features_std = self.mean_std_normalize(features_train)

        real_train_norm = features_train_norm[0:real_train.shape[0], :]

        attack_train_norm = features_train_norm[real_train.shape[0]:, :]

        real_cv_norm, _, _ = self.mean_std_normalize(real_cv, features_mean, features_std)

        attack_cv_norm, _, _ = self.mean_std_normalize(attack_cv, features_mean, features_std)

        return real_train_norm, real_cv_norm, attack_train_norm, attack_cv_norm


    #==========================================================================
    def train_svm(self, training_features, n_samples = 10000,
                  machine_type = 'C_SVC', kernel_type = 'RBF',
                  trainer_grid_search_params = { 'cost': [2**p for p in range(-5, 16, 2)], 'gamma': [2**p for p in range(-15, 4, 2)]},
                  mean_std_norm_flag = False,
                  projector_file = "",
                  save_debug_data_flag = True,
                  reduced_train_data_flag = False,
                  n_train_samples = 50000):
        """
        First, this function tunes the hyper-parameters of the SVM classifier using
        grid search on the sub-sets of training data. Train and cross-validation
        subsets for both classes are formed from the available input training_features.

        Once successfull parameters are determined the SVM is trained on the
        whole training data set. The resulting machine is returned by the function.

        **Parameters:**

        ``training_features`` : [[FrameContainer], [FrameContainer]]
            A list containing two elements: [0] - a list of Frame Containers with
            feature vectors for the real class; [1] - a list of Frame Containers with
            feature vectors for the attack class.

        ``n_samples`` : :py:class:`int`
            Number of uniformly selected feature vectors per class defining the
            sizes of sub-sets used in the hyper-parameter grid search.

        ``machine_type`` : :py:class:`str`
            A type of the SVM machine. Please check ``bob.learn.libsvm`` for
            more details.

        ``kernel_type`` : :py:class:`str`
            A type of kerenel for the SVM machine. Please check ``bob.learn.libsvm``
            for more details.

        ``trainer_grid_search_params`` : :py:class:`dict`
            Dictionary containing the hyper-parameters of the SVM to be tested
            in the grid-search.

        ``mean_std_norm_flag`` : :py:class:`bool`
            Perform mean-std normalization of data if set to True. Default: False.

        ``projector_file`` : :py:class:`str`
            The name of the file to save the trained projector to. Only the path
            of this file is used in this function. The file debug_data.hdf5 will
            be save in this path. This file contains information, which might be
            usefull for debugging.

        ``save_debug_data_flag`` : :py:class:`bool`
            Save the data, which might be usefull for debugging if ``True``.
            Default: ``True``.

        ``reduced_train_data_flag`` : :py:class:`bool`
            Reduce the amount of final training samples if set to ``True``.
            Default: ``False``.

        ``n_train_samples`` : :py:class:`int`
            Number of uniformly selected feature vectors per class defining the
            sizes of sub-sets used in the final traing of the SVM.
            Default: 50000.

        **Returns:**

        ``machine`` : object
            A trained SVM machine.
        """

        # get the data for the hyper-parameter grid-search:
        real_train, real_cv, attack_train, attack_cv = self.prepare_data_for_hyper_param_grid_search(training_features, n_samples)

        if mean_std_norm_flag:
            # normalize the data:
            real_train, real_cv, attack_train, attack_cv = self.norm_train_cv_data(real_train, real_cv, attack_train, attack_cv)

        precisions_cv = [] # for saving the precision on the cross-validation set

        precisions_train = []

        trainer_grid_search_params_list = self.combinations(trainer_grid_search_params) # list containing all combinations of params

        for trainer_grid_search_param in trainer_grid_search_params_list:

            # initialize the SVM trainer:
            trainer = bob.learn.libsvm.Trainer(machine_type = machine_type,
                                               kernel_type = kernel_type,
                                               probability = True)

            for key in trainer_grid_search_param.keys():

                setattr(trainer, key, trainer_grid_search_param[key]) # set the params of trainer

            data  = [np.copy(real_train), np.copy(attack_train)] # data used for training the machine in the grid-search

            machine = trainer.train(data) # train the machine

            precision_cv = self.comp_prediction_precision(machine, np.copy(real_cv), np.copy(attack_cv))

            precision_train = self.comp_prediction_precision(machine, np.copy(real_train), np.copy(attack_train))

            precisions_cv.append(precision_cv)

            precisions_train.append(precision_train)

            del data
            del machine
            del trainer

        selected_params = trainer_grid_search_params_list[np.argmax(precisions_cv)] # best SVM parameters according to CV set

        trainer = bob.learn.libsvm.Trainer(machine_type = machine_type,
                                           kernel_type = kernel_type,
                                           probability = True)

        for key in selected_params.keys():

            setattr(trainer, key, selected_params[key]) # set the params of trainer

        # Save the data, which is usefull for debugging.
        if save_debug_data_flag:

            debug_file = os.path.join( os.path.split(projector_file)[0], "debug_data.hdf5" )
            debug_dict = {}
            debug_dict['precisions_train'] = precisions_train
            debug_dict['precisions_cv'] = precisions_cv
            debug_dict['cost'] = selected_params['cost']
            debug_dict['gamma'] = selected_params['gamma']
            f = bob.io.base.HDF5File(debug_file, 'w') # open hdf5 file to save the debug data
            for key in debug_dict.keys():
                f.set(key, debug_dict[key])
            del f

        # training_features[0] - training features for the REAL class.
        real = self.convert_list_of_frame_cont_to_array(training_features[0]) # output is array
        # training_features[1] - training features for the ATTACK class.
        attack = self.convert_list_of_frame_cont_to_array(training_features[1]) # output is array

        if mean_std_norm_flag:
            # Normalize the data:
            features = np.vstack([real, attack])
            features_norm, features_mean, features_std = self.mean_std_normalize(features)
            real =   features_norm[0:real.shape[0], :] # The array is now normalized
            attack = features_norm[real.shape[0]:, :] # The array is now normalized

        if reduced_train_data_flag:

            # uniformly select subsets of features:
            real = self.select_quasi_uniform_data_subset(real, n_train_samples)
            attack = self.select_quasi_uniform_data_subset(attack, n_train_samples)

        data = [np.copy(real), np.copy(attack)] # data for final training

        machine = trainer.train(data) # train the machine

        if mean_std_norm_flag:
            machine.input_subtract = features_mean # subtract the mean of train data
            machine.input_divide   = features_std  # divide by std of train data

        del data

        return machine


    #==========================================================================
    def train_projector(self, training_features, projector_file):
        """
        Train SVM feature projector and save the trained SVM to a given file.
        The ``requires_projector_training = True`` flag must be set to True to
        enable this function.

        **Parameters:**

        ``training_features`` : [[FrameContainer], [FrameContainer]]
            A list containing two elements: [0] - a list of Frame Containers with
            feature vectors for the real class; [1] - a list of Frame Containers with
            feature vectors for the attack class.

        ``projector_file`` : :py:class:`str`
            The file to save the trained projector to.
            This file should be readable with the :py:meth:`load_projector` function.
        """

        machine = self.train_svm(training_features = training_features,
                                 n_samples = self.n_samples,
                                 machine_type = self.machine_type,
                                 kernel_type = self.kernel_type,
                                 trainer_grid_search_params = self.trainer_grid_search_params,
                                 mean_std_norm_flag = self.mean_std_norm_flag,
                                 projector_file = projector_file,
                                 save_debug_data_flag = self.save_debug_data_flag,
                                 reduced_train_data_flag = self.reduced_train_data_flag,
                                 n_train_samples = self.n_train_samples)

        f = bob.io.base.HDF5File(projector_file, 'w') # open hdf5 file to save to

        machine.save(f) # save the machine and normalization parameters

        del f


    #==========================================================================
    def load_projector(self, projector_file):
        """
        Load the pretrained projector/SVM from file to perform a feature projection.
        This function usually is useful in combination with the
        :py:meth:`train_projector` function.

        Please register `performs_projection = True` in the constructor to
        enable this function.

        **Parameters:**

        ``projector_file`` : :py:class:`str`
            The file to read the projector from.
        """

        f = bob.io.base.HDF5File(projector_file, 'a')

        self.machine = bob.learn.libsvm.Machine(f)

        del f


    #==========================================================================
    def project(self, feature):
        """
        This function computes class probabilities for the input feature using pretrained SVM.
        The feature in this case is a Frame Container with features for each frame.
        The probabilities will be computed and returned for each frame.

        Set ``performs_projection = True`` in the constructor to enable this function.
        It is assured that the :py:meth:`load_projector` was called before the
        ``project`` function is executed.

        **Parameters:**

        ``feature`` : object
            A Frame Container conteining the features of an individual,
            see ``bob.bio.video.utils.FrameContainer``.

        **Returns:**

        ``probabilities`` : 2D :py:class:`numpy.ndarray`
            An array containing class probabilities for each frame.
            First column contains probabilities for each frame being a real class.
            Second column contains probabilities for each frame being an attack class.
            Must be writable with the ``write_feature`` function and
            readable with the ``read_feature`` function.
        """

        features_array = self.convert_frame_cont_to_array(feature)

        probabilities = self.machine.predict_class_and_probabilities(features_array)[1]

        return probabilities


    #==========================================================================
    def score(self, toscore):
        """
        Returns a probability of a sample being a real class.

        **Parameters:**

        ``toscore`` : 2D :py:class:`numpy.ndarray`
            An array containing class probabilities for each frame.
            First column contains probabilities for each frame being a real class.
            Second column contains probabilities for each frame being an attack class.

        **Returns:**

        ``score`` : :py:class:`float`
            or a list of scores containing individual score for each frame.
            A score value for the object ``toscore``.
            A probability of a sample being a real class.
        """

        if self.frame_level_scores_flag:

            score = toscore[:,0] # here score is a list containing scores for each frame

        else:

            score = np.mean(toscore, axis=0)[0] # compute a single score per video

        return score


    #==========================================================================
    def score_for_multiple_projections(self, toscore):
        """
        Returns a list of scores computed by the score method of this class.

        **Parameters:**

        ``toscore`` : 2D :py:class:`numpy.ndarray`
            An array containing scores computed by score() method of this class.

        **Returns:**

        ``list_of_scores`` : [:py:class:`float`]
            A list containing the scores.
        """

        if self.frame_level_scores_flag:

            list_of_scores = self.score(toscore)

        else:

            list_of_scores = [self.score(toscore)]

        return list_of_scores


