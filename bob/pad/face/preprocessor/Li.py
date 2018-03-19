import numpy

import logging
logger = logging.getLogger("bob.pad.face")

from bob.bio.base.preprocessor import Preprocessor

from bob.rppg.base.utils import build_bandpass_filter 
import bob.ip.dlib

from bob.rppg.cvpr14.extract_utils import kp66_to_mask
from bob.rppg.cvpr14.extract_utils import compute_average_colors_mask
from bob.rppg.cvpr14.filter_utils import detrend
from bob.rppg.cvpr14.filter_utils import average


class Li(Preprocessor):
  """
  This class extract the pulse signal from a video sequence.
  
  The pulse is extracted according to Li's CVPR 14 algorithm.
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

  debug: boolean          
    Plot some stuff 
  """
  def __init__(self, indent = 10, lambda_ = 300, window = 3, framerate = 25, bp_order = 32, debug=False, **kwargs):

    super(Li, self).__init__(**kwargs)
    
    self.indent = indent
    self.lambda_ = lambda_
    self.window = window
    self.framerate = framerate
    self.bp_order = bp_order
    self.debug = debug

  def __call__(self, frames, annotations=None):
    """
    Compute the pulse signal for the given frame sequence

    **Parameters:**

    frames: :pyclass: `bob.bio.video.utils.FrameContainer`
      Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
      for further details.

    annotations: :py:class:`dict`
      A dictionary containing annotations of the face bounding box.
      Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col)}``

    **Returns:**

      pulse: numpy.array of size (nb_frame, 3)
        The pulse signal in each color channel (RGB)  
    """
    video = frames.as_array()
    nb_frames = video.shape[0]

    # the meancolor of the face along the sequence
    face_color = numpy.zeros((nb_frames, 3), dtype='float64')

    # build the bandpass filter one and for all
    bandpass_filter = build_bandpass_filter(self.framerate, self.bp_order, plot=False)

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
        # looks like one video from replay mobile is upside down !
        rotated_shape = bob.ip.base.rotated_output_shape(frame, 180)
        frame_rotated = numpy.ndarray(rotated_shape, dtype=numpy.float64)
        from bob.ip.base import rotate
        bob.ip.base.rotate(frame, frame_rotated, 180)
        frame_rotated = frame_rotated.astype(numpy.uint8)
        logger.warning("Rotating again ...")
        try:
          ldms = detector(frame_rotated)
        except TypeError:
          ldms = previous_ldms
          # so do nothing ...
          logger.warning("No mask detected in frame {}".format(i))
          face_color[i] = 0
          continue
        frame = frame_rotated

      if self.debug:
        from matplotlib import pyplot
        display = numpy.copy(frame)
        for p in ldms:
          bob.ip.draw.plus(display, p, radius=5, color=(255, 0, 0))
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(display, 2),2))
        pyplot.show()

      ldms = numpy.array(ldms)
      mask_points, mask = kp66_to_mask(frame, ldms, self.indent, self.debug)
      face_color[i] = compute_average_colors_mask(frame, mask, self.debug)

      previous_ldms = ldms 
      counter += 1

    pulse = numpy.zeros((nb_frames, 3), dtype='float64')
    for i in range(3):
      # detrend
      detrended = detrend(face_color[:, i], self.lambda_)
      # average
      averaged = average(detrended, self.window)
      # bandpass
      from scipy.signal import filtfilt
      pulse[:, i] = filtfilt(bandpass_filter, numpy.array([1]), averaged)

    if self.debug: 
      from matplotlib import pyplot
      for i in range(3):
        f, ax = pyplot.subplots(2, sharex=True)
        ax[0].plot(range(face_color.shape[0]), face_color[:, i], 'g')
        ax[0].set_title('Original color signal')
        ax[1].plot(range(face_color.shape[0]), pulse[:, i], 'g')
        ax[1].set_title('Pulse signal')
        pyplot.show()

    return pulse 
