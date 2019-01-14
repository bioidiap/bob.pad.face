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


class CasiaSurfPadDatabase(PadDatabase): 
    """High level implementation of the Database class for the 3DMAD database.
   
    Note that at the moment, this database only contains a training and validation set.

    The protocol specifies the modality(ies) to load.

    Attributes
    ----------
    db : :py:class:`bob.db.casiasurf.Database`
      the low-level database interface
    low_level_group_names : list of :py:obj:`str`
      the group names in the low-level interface (world, dev, test)
    high_level_group_names : list of :py:obj:`str`
      the group names in the high-level interface (train, dev, eval)

    """
        
    from bob.db.casiasurf import Database as LowLevelDatabase
    self.db = LowLevelDatabase()

    self.low_level_group_names = ('train', 'validation', 'test')  
    self.high_level_group_names = ('train', 'dev', 'eval')

    super(CasiaSurfPadDatabase, self).__init__(
        name='casiasurf',
        protocol=protocol,
        original_directory=original_directory,
        original_extension=original_extension,
        **kwargs)

     @property
    def original_directory(self):
        return self.db.original_directory


    @original_directory.setter
    def original_directory(self, value):
        self.db.original_directory = value

    def objects(self,
                groups=None,
                protocol='all',
                purposes=None,
                model_ids=None,
                **kwargs):
        """Returns a list of CasiaSurfPadFile objects, which fulfill the given restrictions.

        Parameters
        ----------
        groups : list of :py:class:`str`
          The groups of which the clients should be returned.
          Usually, groups are one or more elements of ('train', 'dev', 'eval')
        protocol : :py:class:`str`
          The protocol for which the samples should be retrieved.
        purposes : :py:class:`str`
          The purposes for which Sample objects should be retrieved.
          Usually it is either 'real' or 'attack', but could be 'unknown' as well
        model_ids
          This parameter is not supported in PAD databases yet.

        Returns
        -------
        samples : :py:class:`CasiaSurfPadFilePadFile`
            A list of CasiaSurfPadFile objects.
        """

        groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)

        if groups is not None:
          
          # for training
          lowlevel_purposes = []
          if 'train' in groups and purposes == 'real':
            lowlevel_purposes.append('real') 
          if 'train' in groups and purposes == 'attack':
            lowlevel_purposes.append('attack') 

          # for dev and eval
          if ('dev' in groups or 'test' in groups) and purposes == 'real':
            lowlevel_purposes.append('unknown')
          if ('dev' in groups or 'test' in groups) and purposes == 'attack':
            lowlevel_purposes.append('unknown')

        samples = self.db.objects(sets=groups, purposes=lowlevel_purposes, **kwargs)
        samples = [CasiaSurfPadFile(s) for s in samples]

        return samples

    
    def annotations(self, file):
        """No annotations are provided with this DB
        """
        return None

