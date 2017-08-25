#!/usr/bin/env python

from bob.pad.face.algorithm import VideoCascadeSvmPadAlgorithm


#=======================================================================================
# Define instances here:

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.2}
N = 2
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n2_gamma_02 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.1}
N = 2
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n2_gamma_01 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.05}
N = 2
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n2_gamma_005 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.01}
N = 2
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n2_gamma_001 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)


#=======================================================================================

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.1}
N = 10
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n10_gamma_01 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.05}
N = 10
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n10_gamma_005 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.01}
N = 10
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n10_gamma_001 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.005}
N = 10
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n10_gamma_0005 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)


#=======================================================================================

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.5}
N = 20
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n20_gamma_05 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.2}
N = 20
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n20_gamma_02 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.1}
N = 20
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n20_gamma_01 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.05}
N = 20
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n20_gamma_005 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.01}
N = 20
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n20_gamma_001 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.005}
N = 20
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n20_gamma_0005 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)


MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.001}
N = 20
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = True

algorithm_n20_gamma_0001 = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)


#=======================================================================================

MACHINE_TYPE = 'ONE_CLASS'
KERNEL_TYPE = 'RBF'
SVM_KWARGS = {'nu': 0.001, 'gamma': 0.1}
N = 2
POS_SCORES_SLOPE = 0.01
FRAME_LEVEL_SCORES_FLAG = False

algorithm_n2_gamma_01_video_level = VideoCascadeSvmPadAlgorithm(machine_type = MACHINE_TYPE,
                                        kernel_type = KERNEL_TYPE,
                                        svm_kwargs = SVM_KWARGS,
                                        N = N,
                                        pos_scores_slope = POS_SCORES_SLOPE,
                                        frame_level_scores_flag = FRAME_LEVEL_SCORES_FLAG)


