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

  framerate: int
    The framerate of the video sequence.

  bp_order: int
    The order of the bandpass filter

  debug: boolean          
    Plot some stuff 
  """
  def __init__(self, framerate=25, bp_order=32, debug=False, **kwargs):

    super(PPGSecure, self).__init__(**kwargs)
    
    self.framerate = framerate
    self.bp_order = bp_order
    self.debug = debug
    
    # build the bandpass filter one and for all
    self.bandpass_filter = build_bandpass_filter(self.framerate, self.bp_order, min_freq=0.5, max_freq=5, plot=False)
    
    # landmarks detection
    self.detector = bob.ip.dlib.DlibLandmarkExtraction()

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

    previous_ldms = None
    for i, frame in enumerate(video):

      logger.debug("Processing frame {}/{}".format(i, nb_frames))
      if self.debug:
        from matplotlib import pyplot
        pyplot.imshow(numpy.rollaxis(numpy.rollaxis(frame, 2),2))
        pyplot.show()
     
      # detect landmarks
      try:
        ldms = self.detector(frame)
      except TypeError:
        # looks like one video from replay mobile is upside down !
        rotated_shape = bob.ip.base.rotated_output_shape(frame, 180)
        frame_rotated = numpy.ndarray(rotated_shape, dtype=numpy.float64)
        from bob.ip.base import rotate
        bob.ip.base.rotate(frame, frame_rotated, 180)
        frame_rotated = frame_rotated.astype(numpy.uint8)
        logger.warning("Rotating again ...")
        try:
          ldms = self.detector(frame_rotated)
        except TypeError:
          ldms = previous_ldms
          # so do nothing ...
          logger.warning("No mask detected in frame {}".format(i))
          green_mean[i, :] = 0
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

      # get the mask and the green value in the different ROIs
      masks = self._get_masks(frame, ldms)
      for k in range(5):
        green_mean[i, k] = compute_average_colors_mask(frame, masks[k], self.debug)[1]

      previous_ldms = ldms 

    pulses = numpy.zeros((nb_frames, 5), dtype='float64')
    for k in range(5):
      # substract mean
      green_mean[:, k] = green_mean[:, k] - numpy.mean(green_mean[:, k])
      # bandpass
      from scipy.signal import filtfilt
      pulses[:, k] = filtfilt(self.bandpass_filter, numpy.array([1]), green_mean[:, k])

    if self.debug: 
      from matplotlib import pyplot
      for k in range(5):
        f, ax = pyplot.subplots(2, sharex=True)
        ax[0].plot(range(green_mean.shape[0]), green_mean[:, k], 'g')
        ax[0].set_title('Original color signal')
        ax[1].plot(range(green_mean.shape[0]), pulses[:, k], 'g')
        ax[1].set_title('Pulse signal')
        pyplot.show()

    return pulses 


  def _get_masks(self, image, ldms):
    """ get the 5 masks for rPPG signal extraction

    **Parameters**

    ldms: numpy.array
      The landmarks, as retrieved by bob.ip.dlib.DlibLandmarkExtraction()

    **Returns**
      masks: boolean
        
    """
    masks = []

    # mask 1: forehead
    # defined by 12 points: upper eyebrows (points 18 to 27)
    # plus two additional points:
    # - above 18, at a distance of (18-27)/2
    # - above 27, at a distance of (18-27)/2
    #
    # Note 0 -> y, 1 -> x
    mask_points = []
    for k in range(17, 27):
      mask_points.append([int(ldms[k, 0]), int(ldms[k, 1])])
    above_20_x = int(ldms[19, 1])
    above_20_y = int(ldms[19, 0]) - int(abs(ldms[17, 1] - ldms[26, 1]) / 3)
    above_25_x = int(ldms[24, 1])
    above_25_y = int(ldms[24, 0]) - int(abs(ldms[17, 1] - ldms[26, 1]) / 3)
    mask_points.append([above_25_y, above_25_x])
    mask_points.append([above_20_y, above_20_x])
    masks.append(get_mask(image, mask_points))
    
    # mask 2: right cheek (left-hand side when looking at the screen)
    # defined by points 1-7 + 49 + 32 + 42
    mask_points = []
    for k in range(7):
      mask_points.append([int(ldms[k, 0]), int(ldms[k, 1])])
    mask_points.append([int(ldms[48, 0]), int(ldms[48, 1])])
    mask_points.append([int(ldms[31, 0]), int(ldms[31, 1])])
    mask_points.append([int(ldms[41, 0]), int(ldms[41, 1])])
    masks.append(get_mask(image, mask_points))

    # mask 3: left cheek 
    # defined by points 17-11 + 55 + 36 + 47
    mask_points = []
    for k in range(16, 10, -1):
      mask_points.append([int(ldms[k, 0]), int(ldms[k, 1])])
    mask_points.append([int(ldms[54, 0]), int(ldms[54, 1])])
    mask_points.append([int(ldms[35, 0]), int(ldms[35, 1])])
    mask_points.append([int(ldms[46, 0]), int(ldms[46, 1])])
    masks.append(get_mask(image, mask_points))
   
    # mask 4: right above shoulder
    mask_points = []
    mask_points.append([int(ldms[2, 0]), int(ldms[2, 1] - 10)])
    mask_points.append([int(ldms[2, 0]), int(ldms[2, 1] - 60)])
    mask_points.append([int(ldms[2, 0] + 50), int(ldms[2, 1] - 60)])
    mask_points.append([int(ldms[2, 0] + 50), int(ldms[2, 1] - 10)])
    masks.append(get_mask(image, mask_points))

    # mask 5: left above shoulder
    mask_points = []
    mask_points.append([int(ldms[14, 0]), int(ldms[14, 1] + 10)])
    mask_points.append([int(ldms[14, 0]), int(ldms[14, 1] + 60)])
    mask_points.append([int(ldms[14, 0] + 50), int(ldms[14, 1] + 60)])
    mask_points.append([int(ldms[14, 0] + 50), int(ldms[14, 1] + 10)])
    masks.append(get_mask(image, mask_points))

    return masks


