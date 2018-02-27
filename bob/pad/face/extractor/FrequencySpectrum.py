#!/usr/bin/env python
# encoding: utf-8

import numpy

from bob.bio.base.extractor import Extractor

import logging
logger = logging.getLogger("bob.pad.face")

from scipy.signal import welch


class FrequencySpectrum(Extractor, object):
  """
  Compute the Frequency Spectrum of the given signal.

  The computation is made using Welch's algorithm.

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
  def __init__(self, framerate=25, nsegments=12, nfft=256, debug=False, **kwargs):

    super(FrequencySpectrum, self).__init__()
    
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
    # sanity check
    if signal.ndim == 1:
      if numpy.isnan(numpy.sum(signal)):
        return
    if signal.ndim == 2 and (signal.shape[1] == 3):
      if numpy.isnan(numpy.sum(signal[:, 1])):
        return

    output_dim = int((self.nfft / 2) + 1)
   
    # we have a single pulse signal
    if signal.ndim == 1:
      f, psd = welch(signal, self.framerate, nperseg=self.nsegments, nfft=self.nfft)

    # we have 3 pulse signal (Li's preprocessing)
    # in this case, return the signal corresponding to the green channel
    if signal.ndim == 2 and (signal.shape[1] == 3):
      psds = numpy.zeros((3, output_dim))
      for i in range(3):
        f, psds[i] = welch(signal[:, i], self.framerate, nperseg=self.nsegments, nfft=self.nfft)
      psd = psds[1]
      
    if self.debug: 
      from matplotlib import pyplot
      pyplot.semilogy(f, psd, 'k')
      pyplot.title('Power spectrum of the signal')
      pyplot.show()

    return psd
