#!/usr/bin/env python
# encoding: utf-8

import numpy

from bob.bio.base.extractor import Extractor

import logging
logger = logging.getLogger("bob.pad.face")

from scipy.fftpack import rfft


class LTSS(Extractor, object):
  """
  Compute Long-term spectral statistics of a pulse signal.
  
  The features are described in the following article:
  
    @Article {
    Author         = {Muckenhirn, Hannah and Korshunov, Pavel and
                     Magimai-Doss, Mathew and Marcel, Sebastien }
    Title          = {Long-Term Spectral Statistics for Voice Presentation
                     Attack Detection},
    Journal        = {IEEE/ACM Trans. Audio, Speech and Lang. Proc.},
    Volume         = {25},
    Number         = {11},
    Pages          = {2098--2111},
    year           = 2017
    }

  **Parameters:**

  framerate: int
    The sampling frequency of the signal (i.e the framerate ...) 

  nfft: int
    Number of points to compute the FFT

  debug: boolean
    Plot stuff
  """
  def __init__(self, window_size=25, framerate=25, nfft=64, concat=False, debug=False, **kwargs):

    super(LTSS, self).__init__()
    
    self.framerate = framerate
    self.nfft = nfft
    self.debug = debug
    self.window_size = window_size
    self.concat = concat

  def _get_ltss(self, signal):
    
    # log-magnitude of DFT coefficients
    log_mags = []
    window_stride = int(self.window_size / 2)
   
    # go through windows
    for w in range(0, (signal.shape[0] - self.window_size), window_stride):
      fft = rfft(signal[w:w+self.window_size], n=self.nfft)
      mags = numpy.zeros(int(self.nfft/2), dtype=numpy.float64)
      mags[0] = abs(fft[0])
      index = 1
      for i in range(1, (fft.shape[0]-1), 2):
        mags[index] = numpy.sqrt(fft[i]**2 + fft[i+1]**2)
        if mags[index] < 1:
          mags[index] = 1
        index += 1
      log_mags.append(numpy.log(mags))

    # get the long term statistics
    log_mags = numpy.array(log_mags)
    mean = numpy.mean(log_mags, axis=0)
    std = numpy.std(log_mags, axis=0)
    ltss = numpy.concatenate([mean, std])
    return ltss


  def __call__(self, signal):
    """
    Computes the long-term spectral statistics for a given signal.

    **Parameters**

    signal: numpy.array 
      The signal

    **Returns:**

      feature: numpy.array 
       the long-term spectral statistics feature vector 
    """
    # sanity check
    if signal.ndim == 1:
      if numpy.isnan(numpy.sum(signal)):
        return
    if signal.ndim == 2 and (signal.shape[1] == 3):
      for i in range(signal.shape[1]):
        if numpy.isnan(numpy.sum(signal[:, i])):
          return

    if signal.ndim == 1:
      feature = self._get_ltss(signal)

    if signal.ndim == 2 and (signal.shape[1] == 3):
      
      if not self.concat:
        feature = self._get_ltss(signal[:, 1])
      else:
        ltss = []
        for i in range(signal.shape[1]):
          ltss.append(self._get_ltss(signal[:, i]))
        feature = numpy.concatenate([ltss[0], ltss[1], ltss[2]])

    if numpy.isnan(numpy.sum(feature)):
      logger.warn("Feature not extracted")
      return
    if numpy.sum(feature) == 0:
      logger.warn("Feature not extracted")
      return

    return feature
