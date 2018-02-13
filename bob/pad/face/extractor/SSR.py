#!/usr/bin/env python
# encoding: utf-8

import six
import numpy

import bob.bio.video
from bob.bio.base.extractor import Extractor
from bob.pad.face.extractor import VideoDataLoader

import bob.ip.facedetect
import bob.ip.base
import bob.ip.skincolorfilter

import logging
logger = logging.getLogger("bob.pad.face")


from bob.rppg.base.utils import crop_face

from bob.rppg.ssr.ssr_utils import get_eigen
from bob.rppg.ssr.ssr_utils import plot_eigenvectors
from bob.rppg.ssr.ssr_utils import build_P



class SSR(Extractor, object):
  """
  Extract pulse signal according to the SSR algorithm

  **Parameters:**

  skin_threshold: float
    The threshold for skin color probability

  skin_init: bool
    If you want to re-initailize the skin color distribution at each frame

  stride: int
    The temporal stride. 

  """
  def __init__(self, skin_threshold=0.5, skin_init=False, stride=25, **kwargs):

    super(SSR, self).__init__()

    self.skin_threshold = skin_threshold
    self.skin_init = skin_init
    self.stride = stride

    self.skin_filter = bob.ip.skincolorfilter.SkinColorFilter()


  def __call__(self, frames):
    """
    Compute the pulse signal for the given frame sequence

    **Parameters:**

    frames: FrameContainer or string.
      Video data stored in the FrameContainer,
      see ``bob.bio.video.utils.FrameContainer`` for further details.
      If string, the name of the file to load the video data from is
      defined in it. String is possible only when empty preprocessor is
      used. In this case video data is loaded directly from the database.

    **Returns:**

      pulse: FrameContainer
      Quality Measures for each frame stored in the FrameContainer.
    """

    # load video based on the filename
    assert isinstance(frames, six.string_types)
    if isinstance(frames, six.string_types):
      video_loader = VideoDataLoader()
      video = video_loader(frames)
   
    video = video.as_array()
    nb_frames = video.shape[0]

    # the result -> the pulse signal 
    output_data = numpy.zeros(nb_frames, dtype='float64')

    # store the eigenvalues and the eigenvectors at each frame 
    eigenvalues = numpy.zeros((3, nb_frames), dtype='float64')
    eigenvectors = numpy.zeros((3, 3, nb_frames), dtype='float64')

    ### LET'S GO

    #XXX 
    plot = True

    counter = 0
    previous_bbox = None
    for i, frame in enumerate(video):

      logger.debug("Processing frame %d/%d...", i, nb_frames)
      try:
        bbox, quality = bob.ip.facedetect.detect_single_face(frame)
      except:
        bbox = previous_bbox
        logger.warning("Using bounding box from previous frame ...")

      face = crop_face(frame, bbox, bbox.size[1])

      #from matplotlib import pyplot
      #pyplot.imshow(numpy.rollaxis(numpy.rollaxis(face, 2),2))
      #pyplot.show()

      # skin filter
      if counter == 0 or self.skin_init:
        self.skin_filter.estimate_gaussian_parameters(face)
        logger.debug("Skin color parameters:\nmean\n{0}\ncovariance\n{1}".format(self.skin_filter.mean, self.skin_filter.covariance))
      skin_mask = self.skin_filter.get_skin_mask(face, self.skin_threshold)
      skin_pixels = face[:, skin_mask]

      #from matplotlib import pyplot
      #skin_mask_image = numpy.copy(face)
      #skin_mask_image[:, skin_mask] = 255
      #pyplot.title("skin pixels in frame {0}".format(i))
      #pyplot.imshow(numpy.rollaxis(numpy.rollaxis(skin_mask_image, 2),2))
      #pyplot.show()
  
      skin_pixels = skin_pixels.astype('float64') / 255.0

      # build c matrix and get eigenvectors and eigenvalues
      eigenvalues[:, counter], eigenvectors[:, :, counter] = get_eigen(skin_pixels)
      #plot_eigenvectors(skin_pixels, eigenvectors[:, :, counter])

      # build P and add it to the pulse signal
      if counter >= self.stride:
        tau = counter - self.stride
        p = build_P(counter, self.stride, eigenvectors, eigenvalues)
        output_data[tau:counter] += (p - numpy.mean(p)) 
         
      counter += 1

    # plot the pulse signal
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(range(nb_frames), output_data)
    #plt.show()

    return output_data
