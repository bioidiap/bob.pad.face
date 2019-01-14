#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import bob.io.video
from bob.bio.video import FrameSelector, FrameContainer
from bob.pad.face.database import VideoPadFile  
from bob.pad.base.database import PadDatabase

class CasiaSurfPadFile(VideoPadFile):
    """
    A high level implementation of the File class for the CASIA-SURF database.

    Note that this does not represent a file per se, but rather a sample
    that may contain more than one file.

    Attributes
    ----------
    f : :py:class:`object`
      An instance of the Sample class defined in the low level db interface
      of the CASIA-SURF database, in the bob.db.casiasurf.models.py file.
    
    """

    def __init__(self, f, stream_type):
      """ Init

      Parameters
      ----------
      f : :py:class:`object`
        An instance of the Sample class defined in the low level db interface
        of the CASIA-SURF database, in the bob.db.casiasurf.models.py file.
      stream_type: str of list of str
        The streams to be loaded.
      """
      self.f = f
      self.stream_type = stream_type
      super(CasiaSurfPadFile, self).__init__(
            client_id=f.id,
            attack_type=attack_type)


    def load(self, directory=None, extension='.jpg', frame_selector=FrameSelector(selection_style='all')):
        """Overridden version of the load method defined in ``VideoPadFile``.

        Parameters
        ----------
        directory : :py:class:`str`
          String containing the path to the CASIA-SURF database 
        extension : :py:class:`str`
          Extension of the image files 
        frame_selector : :py:class:`bob.bio.video.FrameSelector`
            The frame selector to use.

        Returns
        -------
        dict:
        "video" data (i.e. one frame) for multiple streams stored in the dictionary. 
        The structure of the dictionary: ``data={"stream1_name" : FrameContainer1, "stream2_name" : ...}``
        Names of the streams are defined in ``self.stream_type``.
        """
        
        # get the dict of numpy array
        data = self.f.load(directory, extension, modality=self.modality)
      
        # convert that to dict of FrameContainer
        data_to_return = {}
        for k in data.keys():
          frame_container = FrameContainer()  
          for idx, item in enumerate(data[k]):
            frame_container.add(idx, item)  # add frame to FrameContainer
          data_to_return[k] = frame_container

        return data_to_return
