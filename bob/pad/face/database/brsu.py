#!/usr/bin/env python
# -*- coding: utf-8 -*-


from bob.pad.face.database import VideoPadFile  
from bob.pad.base.database import PadDatabase

from bob.extension import rc

class BRSUPadFile(VideoPadFile):
    """
    A high level implementation of the File class for the BRSU database.

    Note that this does not represent a file per se, but rather a sample
    that may contain more than one file.

    Attributes
    ----------
    f : :py:class:`object`
      An instance of the Sample class defined in the low level db interface
      of the BRSU database, in the bob.db.brsu.models.py file.
    
    """

    def __init__(self, s):
      """ Init

      Parameters
      ----------
      s : :py:class:`object`
        An instance of the Sample class defined in the low level db interface
        of the BRSU database, in the bob.db.brsu.models.py file.
      """
      self.s = s
      attack_type = str(s.attack_type)

      if attack_type == '0':
        attack_type = None

      super(BRSUPadFile, self).__init__(
            client_id=s.id,
            file_id=s.id,
            attack_type=attack_type,
            path=s.id)
      

    def load(self, directory=rc['bob.db.brsu.directory'], extension=None):
        """Overloaded version of the load method defined in ``VideoPadFile``.

        Parameters
        ----------
        directory : :py:class:`str`
          String containing the path to the BRSU database 
        extension : :py:class:`str`
          Not used here, since a sample contains more than one file,
          possibly with different extensions

        Returns
        -------
        dict:
          image data for multiple streams stored in the dictionary. 
          The structure of the dictionary: ``data={"stream1_name" : numpy array, "stream2_name" : numpy array}``
        """
        return self.s.load(directory)


class BRSUPadDatabase(PadDatabase): 
    """High level implementation of the Database class for the BRSU database.
   
    Attributes
    ----------
    db : :py:class:`bob.db.brsu.Database`
      the low-level database interface

    """
       
    def __init__(self, protocol='test', original_directory=rc['bob.db.brsu.directory'], original_extension=None, **kwargs):
      """Init function

        Parameters
        ----------
        protocol : :py:class:`str`
          The name of the protocol that defines the default experimental setup for this database.
        original_directory : :py:class:`str`
          The directory where the original data of the database are stored.
        original_extension : :py:class:`str`
          The file name extension of the original data.
        
      """

      from bob.db.brsu import Database as LowLevelDatabase
      self.db = LowLevelDatabase()

      super(BRSUPadDatabase, self).__init__(
          name='brsu',
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
                protocol='test',
                purposes=None,
                model_ids=None,
                **kwargs):
        """Returns a list of BRSUPadFile objects, which fulfill the given restrictions.

        Parameters
        ----------
        groups : list of :py:class:`str`
          The groups of which the clients should be returned.
          Usually, groups are one or more elements of ('train', 'dev', 'eval')
        protocol : :py:class:`str`
          The protocol for which the samples should be retrieved.
        purposes : :py:class:`str`
          The purposes for which Sample objects should be retrieved.
          Usually it is either 'real' or 'attack'
        model_ids
          This parameter is not supported in PAD databases yet.

        Returns
        -------
        samples : :py:class:`BRSUPadFile`
            A list of BRSUPadFile objects.
        """
        lowlevel_purposes = None
        if groups is not None and purposes is not None:
          
          # for training
          lowlevel_purposes = []
          if 'train' in groups and 'real' in purposes:
            lowlevel_purposes.append('real') 
          if 'train' in groups and 'attack' in purposes:
            lowlevel_purposes.append('attack') 

          # for eval
          if 'test' in groups and 'real' in purposes:
            lowlevel_purposes.append('real') 
          if 'test' in groups and 'attack' in purposes:
            lowlevel_purposes.append('attack')

        if groups is None and purposes is not None:
          lowlevel_purposes = []
          if 'real' in purposes:
            lowlevel_purposes.append('real')
          if 'attack' in purposes:
            lowlevel_purposes.append('attack')

        samples = self.db.objects(groups=groups, purposes=lowlevel_purposes, **kwargs)
        samples = [BRSUPadFile(s) for s in samples]
        return samples

    
    def annotations(self, file):
        """No annotations are provided with this DB
        """
        return None
