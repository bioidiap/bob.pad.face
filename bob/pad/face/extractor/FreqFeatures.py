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

  nsegments: int
    Number of overlapping segments in Welch procedure

  nfft: int
    Number of points to compute the FFT

  debug: boolean
    Plot stuff
  """
  def __init__(self, framerate=25, nsegments=12, nfft=128, debug=False, **kwargs):

    super(FreqFeatures, self).__init__()
    
    self.framerate = framerate
    self.nsegments = nsegments
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
    assert signal.ndim == 2 and signal.shape[1] == 3, "You should provide 3 pulse signals"
   
    feature = numpy.zeros(6)

    spectrum_dim = int((self.nfft / 2) + 1)
    psds = numpy.zeros((3, spectrum_dim))
    for i in range(3):
      f, psds[i] = welch(signal[:, i], self.framerate, nperseg=self.nsegments, nfft=self.nfft)

    # find the max of the frequency spectrum in the range of interest
    first = numpy.where(f > 0.7)[0]
    last = numpy.where(f < 4)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)
    
    for i in range(3):
      total_power = numpy.sum(psds[i, range_of_interest])
      max_power = numpy.max(psds[i, range_of_interest])
      feature[i] = max_power
      feature[i+3] = max_power / total_power
      print (max_power)
      print (max_power / total_power)

      if self.debug:
        f_max = f[range_of_interest[max_idx]]
        max_idx = numpy.argmax(psds[i, range_of_interest])
        from matplotlib import pyplot
        pyplot.semilogy(f, psds[i], 'k')
        xmax, xmin, ymax, ymin = pyplot.axis()
        pyplot.vlines(f[range_of_interest[max_idx]], ymin, ymax, color='red')
        pyplot.title('Power spectrum of the signal')
        pyplot.show()

    print(feature)
    import sys
    sys.exit()
    return feature

