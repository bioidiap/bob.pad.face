import numpy

import logging
logger = logging.getLogger("bob.pad.face")

from bob.bio.base.preprocessor import Preprocessor

from bob.rppg.base.utils import build_bandpass_filter 
import bob.ip.dlib

from bob.rppg.cvpr14.extract_utils import get_mask
from bob.rppg.cvpr14.extract_utils import compute_average_colors_mask


class PPGSecure(Preprocessor):
  """
  This class extract the pulse signal from a video sequence.
  
  The pulse is extracted according to what is described in 
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

  indent: int
    Indent (in pixels) to apply to keypoints to get the masks.

  framerate: int
    The framerate of the video sequence.

  bp_order: int
    The order of the bandpass filter

  debug: boolean          
    Plot some stuff 
  """
  def __init__(self, indent = 10, framerate = 25, bp_order = 32, debug=False, **kwargs):

    super(PPGSecure, self).__init__(**kwargs)
    
    self.indent = indent
    self.framerate = framerate
    self.bp_order = bp_order
    self.debug = debug

  def __call__(self, frames, annotations):
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

      pulses: numpy.array of size (5, nb_frame)
        The pulse signals from different area of the image 
    """
    video = frames.as_array()
    nb_frames = video.shape[0]

    # the mean of the green color of the different ROIs along the sequence
    green_mean = numpy.zeros((nb_frames, 5), dtype='float64')

    # build the bandpass filter one and for all
    bandpass_filter = build_bandpass_filter(self.framerate, self.bp_order, min_freq=0.5, max_freq=5, plot=False)

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

      masks = self._get_masks(ldms)
      import sys
      sys.exit()
      #for i in range(5):
      #  face_color[i] = compute_average_colors_mask(frame, mask, self.debug)

      #previous_ldms = ldms 
      #counter += 1

    #pulse = numpy.zeros((nb_frames, 3), dtype='float64')
    #for i in range(3):
    #  # detrend
    #  detrended = detrend(face_color[:, i], self.lambda_)
    #  # average
    #  averaged = average(detrended, self.window)
    #  # bandpass
    #  from scipy.signal import filtfilt
    #  pulse[:, i] = filtfilt(bandpass_filter, numpy.array([1]), averaged)

    #if self.debug: 
    #  from matplotlib import pyplot
    #  for i in range(3):
    #    f, ax = pyplot.subplots(2, sharex=True)
    #    ax[0].plot(range(face_color.shape[0]), face_color[:, i], 'g')
    #    ax[0].set_title('Original color signal')
    #    ax[1].plot(range(face_color.shape[0]), pulse[:, i], 'g')
    #    ax[1].set_title('Pulse signal')
    #    pyplot.show()

    #return pulse 

  def _get_masks(ldms):
    """ get the 5 masks for rPPG signal extraction

    **Parameters**

    ldms: numpy.array
      The landmarks, as retrieved by bob.ip.dlib.DlibLandmarkExtraction()

    **Returns**
      masks: boolean
        
    """

    # mask 1: forehead
    # defined by 12 points: upper eyebrows (points 18 to 27)
    # plus two additional points:
    # - above 18, at a distance of (18-27)/2
    # - above 27, at a distance of (18-27)/2
    print(ldms)
    print(ldms.shape)
    mask_points = []
    for i in range(17, 28):
      mask_points.append([int(keypoints[k, 0]), int(keypoints[k, 1])]))


