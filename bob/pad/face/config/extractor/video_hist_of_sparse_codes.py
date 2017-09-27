#!/usr/bin/env python

from bob.pad.face.extractor import VideoHistOfSparseCodes


#=======================================================================================
# Define instances here:

METHOD = "mean"
extractor_mean = VideoHistOfSparseCodes(method = METHOD)


METHOD = "hist"
extractor_hist = VideoHistOfSparseCodes(method = METHOD)


