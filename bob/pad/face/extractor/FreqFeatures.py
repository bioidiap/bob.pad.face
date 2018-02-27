#!/usr/bin/env python
# encoding: utf-8

import numpy

from bob.bio.base.extractor import Extractor

import logging
logger = logging.getLogger("bob.pad.face")

from scipy.signal import welch


class FreqFeatures(Extractor, object):
  """
  Compute features from pulse signals in the three color channels.

  The features are described in the following article:
  
    @InProceedings{li-icpr-2016,
    Author         = {Li, X. and Komulainen, J. and Zhao, G. and Yuen, P-C.
                     and Pietik\"ainen, M.},
    Title          = {Generalized {F}ace {A}nti-spoofing by {D}etecting
                     {P}ulse {F}rom {F}ace {V}ideos},
    BookTitle      = {Intl {C}onf. on {P}attern {R}ecognition ({ICPR})},
    Pages          = {4244-4249},
    year           = 2016
    }


  **Parameters:**

  framerate: int
    The sampling frequency of the signal (i.e the framerate ...) 

  nfft: int
    Number of points to compute the FFT

  debug: boolean
    Plot stuff
  """
  def __init__(self, framerate=25, nfft=512, debug=False, **kwargs):

    super(FreqFeatures, self).__init__()
    
    self.framerate = framerate
    self.nfft = nfft
    self.debug = debug

  def __call__(self, signal):
    """
    Compute the frequency spectrum for the given signal.

    **Parameters:**

    signal: numpy.array 
      The signal

    **Returns:**

      freq: numpy.array 
       the frequency spectrum 
    """
    # sanity check
    assert signal.ndim == 2 and signal.shape[1] == 3, "You should provide 3 pulse signals"
    for i in range(3):
      if numpy.isnan(numpy.sum(signal[:, i])):
        return
    
    feature = numpy.zeros(6)
    
    # when keypoints have not been detected, the pulse is zero everywhere
    # hence, no psd and no features
    zero_counter = 0
    for i in range(3):
      if numpy.sum(signal[:, i]) == 0:
        zero_counter += 1
    if zero_counter == 3:
      logger.warn("Feature is all zeros")
      return feature

    # get the frequency spectrum
    spectrum_dim = int((self.nfft / 2) + 1)
    ffts = numpy.zeros((3, spectrum_dim))
    f = numpy.fft.fftfreq(self.nfft) * self.framerate
    f = abs(f[:spectrum_dim])
    for i in range(3):
      ffts[i] = abs(numpy.fft.rfft(signal[:, i], n=self.nfft))
    
    # find the max of the frequency spectrum in the range of interest
    first = numpy.where(f > 0.7)[0]
    last = numpy.where(f < 4)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    # build the feature vector
    for i in range(3):
      total_power = numpy.sum(ffts[i, range_of_interest])
      max_power = numpy.max(ffts[i, range_of_interest])
      feature[i] = max_power
      if total_power == 0:
        feature[i+3] = 0 
      else:
        feature[i+3] = max_power / total_power
   
    # plot stuff, if asked for
    if self.debug:
      from matplotlib import pyplot
      for i in range(3):
        max_idx = numpy.argmax(ffts[i, range_of_interest])
        f_max = f[range_of_interest[max_idx]]
        logger.debug("Inferred HR = {}".format(f_max*60))
        pyplot.plot(f, ffts[i], 'k')
        xmax, xmin, ymax, ymin = pyplot.axis()
        pyplot.vlines(f[range_of_interest[max_idx]], ymin, ymax, color='red')
        pyplot.vlines(f[first_index], ymin, ymax, color='green')
        pyplot.vlines(f[last_index], ymin, ymax, color='green')
        pyplot.show()
    

    if numpy.isnan(numpy.sum(feature)):
      logger.warn("Feature not extracted")
      return
    
    return feature

