#!/usr/bin/env python
# encoding: utf-8

import numpy

from bob.bio.base.extractor import Extractor

import logging
logger = logging.getLogger("bob.pad.face")


class PPGSecure(Extractor, object):
  """
  This class extract the frequency features from pulse signals.
  
  The feature are extracted according to what is described in 
  the following article:

    @InProceedings{nowara-afgr-2017,
      Author         = {E. M. Nowara and A. Sabharwal and A. Veeraraghavan},
      Title          = {P{PGS}ecure: {B}iometric {P}resentation {A}ttack
                       {D}etection {U}sing {P}hotopletysmograms},
      BookTitle      = {I{EEE} {I}ntl {C}onf on {A}utomatic {F}ace and
                       {G}esture {R}ecognition ({AFGR})},
      Volume         = {},
      Number         = {},
      Pages          = {56-62},
      issn           = {},
      seq-number     = {69},
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
  def __init__(self, framerate=25, nfft=32, debug=False, **kwargs):

    super(PPGSecure, self).__init__(**kwargs)
    
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
    assert signal.shape[1] == 5, "You should provide 5 pulses"
    if numpy.isnan(numpy.sum(signal)):
      return

    output_dim = int((self.nfft / 2) + 1)
    
    # get the frequencies
    f = numpy.fft.fftfreq(self.nfft) * self.framerate
   
    # we have 5 pulse signal (Li's preprocessing)
    ffts = numpy.zeros((5, output_dim))
    for i in range(5):
      ffts[i] = abs(numpy.fft.rfft(signal[:, i], n=self.nfft))

    fft = numpy.concatenate([ffts[0], ffts[1], ffts[2], ffts[3], ffts[4]])
      
    if self.debug: 
      from matplotlib import pyplot
      pyplot.plot(range(output_dim*5), fft, 'k')
      pyplot.title('Concatenation of spectra')
      pyplot.show()

    if numpy.isnan(numpy.sum(fft)):
      logger.warn("Feature not extracted")
      return
    if numpy.sum(fft) == 0:
      logger.warn("Feature not extracted")
      return


    return fft
