import bob.io.base
import bob.io.video
from bob.bio.base.preprocessor import Preprocessor

class CopyVideo(Preprocessor):
  """
  Dummy class to load a video, and then write it as a bob.bio.video FrameContainer
  
  Used mainly with the Replay Mobile databases, where the low-level db interface
  takes care of properly rotating the video
  """
  def __init__(self, **kwargs):
    super(CopyVideo, self).__init__(**kwargs)

  def __call__(self, frames, annotations):
    """
    Just returns the video sequence

    **Parameters**

      ``frames`` : FrameContainer
        Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
        for further details.

      ``annotations`` : :py:class:`dict`
          A dictionary containing the annotations for each frame in the video.
          Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
          Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
          is the dictionary defining the coordinates of the face bounding box in frame N.

    **Returns:**

      ``frames`` : FrameContainer
      The input frames, stored in the FrameContainer.
    """
    return frames 

  def write_data(self, frames, filename):
    """
    Writes the given data (that has been generated using the __call__ function of this class) to file.
    This method overwrites the write_data() method of the Preprocessor class.

      **Parameters:**

      ``frames`` :
        data returned by the __call__ method of the class.

      ``filename`` : :py:class:`str`
        name of the file.
    """
    if frames: 
      bob.bio.video.preprocessor.Wrapper(Preprocessor()).write_data(frames, filename)

  def read_data(self, filename):
    """
    Reads the preprocessed data from file.
    This method overwrites the read_data() method of the Preprocessor class.

      **Parameters:**

      ``file_name`` : :py:class:`str`
        name of the file.

      **Returns:**

      ``frames`` : :py:class:`bob.bio.video.FrameContainer`
        Frames stored in the frame container.
    """
    frames = bob.bio.video.preprocessor.Wrapper(Preprocessor()).read_data(filename)
    return frames
