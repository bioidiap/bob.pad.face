#!/usr/bin/env python

from bob.pad.face.extractor import VideoQualityMeasure


#=======================================================================================
# Define instances here:

galbally=True
msu=True
dtype=None

video_quality_measure_galbally_msu = VideoQualityMeasure(galbally=galbally,
                                                         msu=msu,
                                                         dtype=dtype)
