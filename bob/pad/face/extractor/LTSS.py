#!/usr/bin/env python
# encoding: utf-8

import numpy

from bob.bio.base.extractor import Extractor

from bob.core.log import setup
logger = setup("bob.pad.face")

from scipy.fftpack import rfft

class LTSS(Extractor, object):
  """Compute Long-term spectral statistics of a pulse signal.
  
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

  Attributes
  ----------
  framerate: int
    The sampling frequency of the signal (i.e the framerate ...) 
  nfft: int
    Number of points to compute the FFT
  debug: bool
    Plot stuff
  concat: bool
    Flag if you would like to concatenate features from the 3 color channels
  time: int
    The length of the signal to consider (in seconds)
  
  """
  def __init__(self, window_size=25, framerate=25, nfft=64, concat=False, debug=False, time=0, **kwargs):
    """Init function

    Parameters
    ----------
    window_size: int
      The size of the window where FFT is computed
    framerate: int
      The sampling frequency of the signal (i.e the framerate ...) 
    nfft: int
      Number of points to compute the FFT
    concat: bool
      Flag if you would like to concatenate features from the 3 color channels
    debug: bool
      Plot stuff
    time: int
      The length of the signal to consider (in seconds)

    """
    super(LTSS, self).__init__()
    self.framerate = framerate
    self.nfft = nfft
    self.debug = debug
    self.window_size = window_size
    self.concat = concat
    self.time = time

  def _get_ltss(self, signal):
    """Compute long term spectral statistics for  a signal

    Parameters
    ----------
    signal: numpy.ndarray
      The signal

    Returns
    -------
    ltss: numpy.ndarray
      The spectral statistics of the signal.

    """
    window_stride = int(self.window_size / 2)

    # log-magnitude of DFT coefficients
    log_mags = []
   
    # go through windows
    for w in range(0, (signal.shape[0] - self.window_size), window_stride):
      fft = rfft(signal[w:w+self.window_size], n=self.nfft)
      mags = numpy.zeros(int(self.nfft/2), dtype=numpy.float64)
      
      # XXX : bug was here (no clipping)
      if abs(fft[0]) < 1:
        mags[0] = 1
      else:
        mags[0] = abs(fft[0])
      # XXX 

      index = 1
      for i in range(1, (fft.shape[0]-1), 2):
        mags[index] = numpy.sqrt(fft[i]**2 + fft[i+1]**2)
        if mags[index] < 1:
          mags[index] = 1
        index += 1
      log_mags.append(numpy.log(mags))

    log_mags = numpy.array(log_mags)
    mean = numpy.mean(log_mags, axis=0)
    std = numpy.std(log_mags, axis=0)
    ltss = numpy.concatenate([mean, std])
    return ltss


  def __call__(self, signal):
    """Computes the long-term spectral statistics for given pulse signals.

    Parameters
    ----------
    signal: numpy.ndarray 
      The signal

    Returns
    -------
    feature: numpy.ndarray 
     the computed LTSS features 

    """
    # sanity check
    if signal.ndim == 1:
      if numpy.isnan(numpy.sum(signal)):
        return
    if signal.ndim == 2 and (signal.shape[1] == 3):
      for i in range(signal.shape[1]):
        if numpy.isnan(numpy.sum(signal[:, i])):
          return

    # truncate the signal according to time
    if self.time > 0:
      number_of_frames = self.time * self.framerate
      
      # check that the truncated signal is not longer 
      # than the original one
      if number_of_frames < signal.shape[0]:

        if signal.ndim == 1:
         signal = signal[:number_of_frames]
        if signal.ndim == 2 and (signal.shape[1] == 3):
          new_signal = numpy.zeros((number_of_frames, 3))
          for i in range(signal.shape[1]):
            new_signal[:, i] = signal[:number_of_frames, i]
          signal = new_signal
      else:
        logger.warning("Sequence should be truncated to {}, but only contains {} => keeping original one".format(number_of_frames, signal.shape[0]))

      # also, be sure that the window_size is not bigger that the signal
      if self.window_size > int(signal.shape[0] / 2):
        self.window_size = int(signal.shape[0] / 2)
        logger.warning("Window size reduced to {}".format(self.window_size))

    # we have a single pulse
    if signal.ndim == 1:
      feature = self._get_ltss(signal)

    # pulse for the 3 color channels
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
