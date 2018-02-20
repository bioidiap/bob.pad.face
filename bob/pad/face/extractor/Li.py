#!/usr/bin/env python
# encoding: utf-8

import six
import numpy

import bob.bio.video
from bob.bio.base.extractor import Extractor
from bob.pad.face.extractor import VideoDataLoader

import bob.ip.base
import bob.ip.dlib
import bob.ip.draw

import logging
logger = logging.getLogger("bob.pad.face")

from bob.rppg.cvpr14.extract_utils import kp66_to_mask
from bob.rppg.cvpr14.extract_utils import compute_average_colors_mask
from bob.rppg.cvpr14.filter_utils import detrend
from bob.rppg.cvpr14.filter_utils import average
from bob.rppg.base.utils import build_bandpass_filter 


class Li(Extractor, object):
  """
  Extract pulse signal according to Li's CVPR 14 algorithm.
  Note that this is a simplified version of the original 
  pulse extraction algorithms (mask detection in each 
  frame instead of tranking, no illumination correction,
  no motion pruning)

  **Parameters:**

  indent: int
    Indent (in percent of the face width) to apply to keypoints to get the mask.

  lamda_: int
    the lamba value of the detrend filter

  window: int
    The size of the window of the average filter 

  framerate: int
    The framerate of the video sequence.

  bp_order: int
    The order of the bandpass filter
  """
  def __init__(self, indent = 10, lambda_ = 300, window = 3, framerate = 25, bp_order = 32, debug=False, **kwargs):

    super(Li, self).__init__()
    
    self.indent = indent
    self.lambda_ = lambda_
    self.window = window
    self.framerate = framerate
    self.bp_order = bp_order
    self.debug = debug

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
      and not using any high or low-level db packages (so beware).

    **Returns:**

      pulse: numpy.array 
        The pulse signal 
    """
    if isinstance(frames, six.string_types):
      video_loader = VideoDataLoader()
      video = video_loader(frames)
    else:
      video = frames

    video = video.as_array()
    nb_frames = video.shape[0]

    # the mean green color of the face along the sequence
    face_color = numpy.zeros(nb_frames, dtype='float64')

    # build the bandpass filter one and for all
    bandpass_filter = build_bandpass_filter(self.framerate, self.bp_order, False)

    # landmarks detection
    detector = bob.ip.dlib.DlibLandmarkExtraction()

    counter = 0
    previous_ldms = None
    for i, frame in enumerate(video):

      logger.debug("Processing frame {}/{}".format(counter, nb_frames))
      if self.debug:
        from matplotlib import pyplot
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(frame, 2),2))
        pyplot.show()
     
      # detect landmarks
      try:
        ldms = detector(frame)
      except TypeError:
        
        ####################
        # looks like some images from replay mobile are upside down !
        # looks like bob.ip.rotate has a color issue
        ####################
        
        #rotated_shape = bob.ip.base.rotated_output_shape(frame, 180)
        #frame_rotated = numpy.ndarray(rotated_shape, dtype=numpy.float64)
        #from bob.ip.base import rotate
        #bob.ip.base.rotate(frame, frame_rotated, 180)
        #logger.warning("Rotating again ...")
        #try:
        #  ldms = detector(frame_rotated)
        #except TypeError:
        #  ldms = previous_ldms
        #frame = frame_rotated
        
        # so do nothing ...
        logger.warning("No mask detected in frame {}".format(i))
        face_color[i] = 0
        continue

      if self.debug:
        from matplotlib import pyplot
        display = numpy.copy(frame)
        for p in ldms:
          bob.ip.draw.plus(display, p, radius=5, color=(255, 0, 0))
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(display, 2),2))
        pyplot.show()

      ldms = numpy.array(ldms)
      mask_points, mask = kp66_to_mask(frame, ldms, self.indent, False)
      face_color[i] = compute_average_colors_mask(frame, mask, False)
      
      previous_ldms = ldms 
      counter += 1

    # detrend
    detrended = detrend(face_color, self.lambda_)
    # average
    averaged = average(detrended, self.window)
    # bandpass
    from scipy.signal import filtfilt
    bandpassed = filtfilt(bandpass_filter, numpy.array([1]), averaged)

    if self.debug: 
      from matplotlib import pyplot
      f, ax = pyplot.subplots(4, sharex=True)
      ax[0].plot(range(face_color.shape[0]), face_color, 'g')
      ax[0].set_title('Original signal')
      ax[1].plot(range(face_color.shape[0]), detrended, 'g')
      ax[1].set_title('After detrending')
      ax[2].plot(range(face_color.shape[0]), averaged, 'g')
      ax[2].set_title('After averaging')
      ax[3].plot(range(face_color.shape[0]), bandpassed, 'g')
      ax[3].set_title('Bandpassed signal')
      pyplot.show()
 
    return bandpassed 
