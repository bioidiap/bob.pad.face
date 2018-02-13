#!/usr/bin/env python

from bob.pad.face.extractor import VideoLBPHistogram


#=======================================================================================
# Define instances here:

lbptype='uniform'
elbptype='regular'
rad=1
neighbors=8
circ=False
dtype=None

video_lbp_histogram_extractor_n8r1_uniform = VideoLBPHistogram(lbptype=lbptype,
                                                               elbptype=elbptype,
                                                               rad=rad,
                                                               neighbors=neighbors,
                                                               circ=circ,
                                                               dtype=dtype)
