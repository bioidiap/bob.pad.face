#!/usr/bin/env python

from bob.pad.face.extractor import LBPHistogram

from bob.bio.video.extractor import Wrapper

# =======================================================================================
# Define instances here:

LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None

video_lbp_histogram_extractor_n8r1_uniform = Wrapper(LBPHistogram(
    lbptype=LBPTYPE,
    elbptype=ELBPTYPE,
    rad=RAD,
    neighbors=NEIGHBORS,
    circ=CIRC,
    dtype=DTYPE))


# =======================================================================================
LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None
N_HOR = 1
N_VERT = 1
lbp_histogram_n8r1_uniform_1x1 = Wrapper(LBPHistogram(
                                         lbptype=LBPTYPE,
                                         elbptype=ELBPTYPE,
                                         rad=RAD,
                                         neighbors=NEIGHBORS,
                                         circ=CIRC,
                                         dtype=DTYPE,
                                         n_hor=N_HOR,
                                         n_vert=N_VERT))

ELBPTYPE = 'modified'
lbp_histogram_n8r1_modified_1x1 = Wrapper(LBPHistogram(
                                          lbptype=LBPTYPE,
                                          elbptype=ELBPTYPE,
                                          rad=RAD,
                                          neighbors=NEIGHBORS,
                                          circ=CIRC,
                                          dtype=DTYPE,
                                          n_hor=N_HOR,
                                          n_vert=N_VERT))


# =======================================================================================
LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None
N_HOR = 2
N_VERT = 2
lbp_histogram_n8r1_uniform_2x2 = Wrapper(LBPHistogram(
                                         lbptype=LBPTYPE,
                                         elbptype=ELBPTYPE,
                                         rad=RAD,
                                         neighbors=NEIGHBORS,
                                         circ=CIRC,
                                         dtype=DTYPE,
                                         n_hor=N_HOR,
                                         n_vert=N_VERT))

ELBPTYPE = 'modified'
lbp_histogram_n8r1_modified_2x2 = Wrapper(LBPHistogram(
                                          lbptype=LBPTYPE,
                                          elbptype=ELBPTYPE,
                                          rad=RAD,
                                          neighbors=NEIGHBORS,
                                          circ=CIRC,
                                          dtype=DTYPE,
                                          n_hor=N_HOR,
                                          n_vert=N_VERT))


# =======================================================================================
LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None
N_HOR = 3
N_VERT = 3
lbp_histogram_n8r1_uniform_3x3 = Wrapper(LBPHistogram(
                                         lbptype=LBPTYPE,
                                         elbptype=ELBPTYPE,
                                         rad=RAD,
                                         neighbors=NEIGHBORS,
                                         circ=CIRC,
                                         dtype=DTYPE,
                                         n_hor=N_HOR,
                                         n_vert=N_VERT))

ELBPTYPE = 'modified'
lbp_histogram_n8r1_modified_3x3 = Wrapper(LBPHistogram(
                                          lbptype=LBPTYPE,
                                          elbptype=ELBPTYPE,
                                          rad=RAD,
                                          neighbors=NEIGHBORS,
                                          circ=CIRC,
                                          dtype=DTYPE,
                                          n_hor=N_HOR,
                                          n_vert=N_VERT))


# =======================================================================================
LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None
N_HOR = 4
N_VERT = 4
lbp_histogram_n8r1_uniform_4x4 = Wrapper(LBPHistogram(
                                         lbptype=LBPTYPE,
                                         elbptype=ELBPTYPE,
                                         rad=RAD,
                                         neighbors=NEIGHBORS,
                                         circ=CIRC,
                                         dtype=DTYPE,
                                         n_hor=N_HOR,
                                         n_vert=N_VERT))

ELBPTYPE = 'modified'
lbp_histogram_n8r1_modified_4x4 = Wrapper(LBPHistogram(
                                          lbptype=LBPTYPE,
                                          elbptype=ELBPTYPE,
                                          rad=RAD,
                                          neighbors=NEIGHBORS,
                                          circ=CIRC,
                                          dtype=DTYPE,
                                          n_hor=N_HOR,
                                          n_vert=N_VERT))


# =======================================================================================
LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None
N_HOR = 2
N_VERT = 3
lbp_histogram_n8r1_uniform_3x2 = Wrapper(LBPHistogram(
                                         lbptype=LBPTYPE,
                                         elbptype=ELBPTYPE,
                                         rad=RAD,
                                         neighbors=NEIGHBORS,
                                         circ=CIRC,
                                         dtype=DTYPE,
                                         n_hor=N_HOR,
                                         n_vert=N_VERT))

ELBPTYPE = 'modified'
lbp_histogram_n8r1_modified_3x2 = Wrapper(LBPHistogram(
                                          lbptype=LBPTYPE,
                                          elbptype=ELBPTYPE,
                                          rad=RAD,
                                          neighbors=NEIGHBORS,
                                          circ=CIRC,
                                          dtype=DTYPE,
                                          n_hor=N_HOR,
                                          n_vert=N_VERT))


# =======================================================================================
LBPTYPE = 'uniform'
ELBPTYPE = 'regular'
RAD = 1
NEIGHBORS = 8
CIRC = False
DTYPE = None
N_HOR = 2
N_VERT = 4
lbp_histogram_n8r1_uniform_4x2 = Wrapper(LBPHistogram(
                                         lbptype=LBPTYPE,
                                         elbptype=ELBPTYPE,
                                         rad=RAD,
                                         neighbors=NEIGHBORS,
                                         circ=CIRC,
                                         dtype=DTYPE,
                                         n_hor=N_HOR,
                                         n_vert=N_VERT))

ELBPTYPE = 'modified'
lbp_histogram_n8r1_modified_4x2 = Wrapper(LBPHistogram(
                                          lbptype=LBPTYPE,
                                          elbptype=ELBPTYPE,
                                          rad=RAD,
                                          neighbors=NEIGHBORS,
                                          circ=CIRC,
                                          dtype=DTYPE,
                                          n_hor=N_HOR,
                                          n_vert=N_VERT))
