#!/usr/bin/env python

from bob.pad.face.extractor import VideoLBPHistogram


#=======================================================================================
# Define instances here:

lbptype='regular'
elbptype='regular'
rad=3
neighbors=8
circ=False
dtype=None

video_lbp_histogram_extractor_n8r3 = VideoLBPHistogram(lbptype=lbptype,
                                                       elbptype=elbptype,
                                                       rad=rad,
                                                       neighbors=neighbors,
                                                       circ=circ,
                                                       dtype=dtype)

