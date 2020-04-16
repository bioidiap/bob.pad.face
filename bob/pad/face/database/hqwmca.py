#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import bob.io.base
from bob.pad.base.database import PadDatabase, PadFile
from bob.extension import rc
from bob.pad.face.preprocessor.FaceCropAlign import detect_face_landmarks_in_image

from bob.db.hqwmca.attack_dictionaries import idiap_type_id_config, idiap_subtype_id_config
from bob.io.stream import Stream, StreamFile

import cv2
import bob.io.image
import numpy as np

class HQWMCAPadFile(PadFile):
    """
    A high level implementation of the File class for the HQWMCA database.

    Attributes
    ----------
    vf : :py:class:`object`
      An instance of the VideoFile class defined in the low level db interface
      of the HQWMCA database, in the bob.db.hqwmca.models.py file.
    streams: :py:dict:
      Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file
    
    """

    def __init__(self, vf, streams=None, n_frames=10):
      """ Init

      Parameters
      ----------
      vf : :py:class:`object`
        An instance of the VideoFile class defined in the low level db interface
        of the HQWMCA database, in the bob.db.hqwmca.models.py file.
      streams: :py:dict:
        Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file
      n_frames: int:
        The number of frames, evenly spread, you would like to retrieve 
      
      """
      self.vf = vf
      self.streams = streams
      self.n_frames = n_frames
      attack_type = str(vf.type_id)

      if vf.is_attack():
        pai_desc = idiap_type_id_config[str(vf.type_id)]
        attack_type = 'attack/' + pai_desc
      else:
        attack_type = None

      super(HQWMCAPadFile, self).__init__(
            client_id=vf.client_id,
            file_id=vf.id,
            attack_type=attack_type,
            path=vf.path)
      

    def load(self, directory=rc['bob.db.hqwmca.directory'], extension='.h5'):
        """ Loads data from the given file

        Parameters
        ----------
        directory : :py:class:`str`
          String containing the path to the HQWMCA database 
        extension : :py:class:`str`
          Typical extension of a VideoFile

        Returns
        -------
        
        """
        return self.vf.load(directory, extension, streams=self.streams, n_frames=self.n_frames)


class HQWMCAPadDatabase(PadDatabase): 
    """High level implementation of the Database class for the HQWMCA database.
   
    Attributes
    ----------
    db : :py:class:`bob.db.hqwmca.Database`
      the low-level database interface
    streams: :py:dict:
      Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file

    """
       
    def __init__(self, protocol='grand_test', original_directory=rc['bob.db.hqwmca.directory'], 
                 original_extension='.h5', annotations_dir=None, streams=None, n_frames=10, use_curated_file_list=False, **kwargs):
      """Init function

        Parameters
        ----------
        protocol : :py:class:`str`
          The name of the protocol that defines the default experimental setup for this database.
        original_directory : :py:class:`str`
          The directory where the original data of the database are stored.
        original_extension : :py:class:`str`
          The file name extension of the original data.
        annotations_dir: str
          Path to the annotations
        streams: :py:dict:
          Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file
        n_frames: int:
          The number of frames, evenly spread, you would like to retrieve 
        use_curated_file_list: bool
          Whether to remove all light makeup, unisex glasses and wigs, which are border case attacks, to create a clean set of attacks
          Removes these attacks from all folds. This can either be set as argument or as additional '-curated' in the protocol name.
        
      """
      from bob.db.hqwmca import Database as LowLevelDatabase
      self.db = LowLevelDatabase()
      self.streams = streams
      self.n_frames = n_frames
      self.annotations_dir = annotations_dir
      self.use_curated_file_list = use_curated_file_list

      super(HQWMCAPadDatabase, self).__init__(
          name='hqwmca',
          protocol=protocol,
          original_directory=original_directory,
          original_extension=original_extension)

      self.low_level_group_names = ('train', 'validation', 'test')
      self.high_level_group_names = ('train', 'dev', 'eval') 


    @property
    def original_directory(self):
        return self.db.original_directory


    @original_directory.setter
    def original_directory(self, value):
        self.db.original_directory = value

    def objects(self,
                groups=None,
                protocol=None,
                purposes=None,
                model_ids=None,
                attack_types=None,
                **kwargs):
        """Returns a list of HQWMCAPadFile objects, which fulfill the given restrictions.

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
        attack_types: list of :py:class:`str`
          The attacks you would like to load.

        Returns
        -------
        samples : :py:class:`HQWMCAPadFile`
            A list of HQWMCAPadFile objects.
        """

        if groups is None:
            groups = self.high_level_group_names

        if purposes is None:
            purposes = ['real', 'attack']

        groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)

        if not isinstance(groups, list) and groups is not None and groups is not str: 
          groups = list(groups)

        
        if len(protocol.split('-'))>1 and protocol.split('-')[-1]=='curated':
          self.use_curated_file_list=True

        protocol=protocol.split('-')[0]

        files = self.db.objects(protocol=protocol,
                                groups=groups,
                                purposes=purposes,
                                attacks=attack_types,
                                **kwargs)


        
        if self.use_curated_file_list:
          # Remove Wigs
          files = [f for f in files if 'Wig' not in idiap_subtype_id_config[str(f.type_id)][str(f.subtype_id)]]
          # Remove Make up Level 0
          files = [f for f in files if 'level 0' not in idiap_subtype_id_config[str(f.type_id)][str(f.subtype_id)]]
          # Remove Unisex glasses
          files = [f for f in files if 'Unisex glasses' not in idiap_subtype_id_config[str(f.type_id)][str(f.subtype_id)]]




        return [HQWMCAPadFile(f, self.streams, self.n_frames) for f in files]




    def annotations(self, ff):
        """
        Computes annotations for a given file object ``f``, which
        is an instance of the ``BatlPadFile`` class.

        NOTE: you can pre-compute annotation in your first experiment
        and then reuse them in other experiments setting
        ``self.annotations_temp_dir`` path of this class, where
        precomputed annotations will be saved.

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of ``BatlPadFile`` defined above.

        **Returns:**

        ``annotations`` : :py:class:`dict`
            A dictionary containing annotations for
            each frame in the video.
            Dictionary structure:
            ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where
            ``frameN_dict`` contains coordinates of the
            face bounding box and landmarks in frame N.
        """

        file_path = os.path.join(self.annotations_dir, ff.path + ".json")

        streams=self.streams


        rep_annotations={}


        if os.path.isfile(file_path):  # only if the annotations exists



          with open(file_path, 'r') as json_file:


            annotations = json.load(json_file)

          if not annotations:  # if dictionary is empty

              return None


          # set annotations in source channel

          #video = ff.vf.load(  directory=self.original_directory, extension=self.original_extension, streams=streams, n_frames=self.n_frames)['color']

          color_stream = streams['color']
          rep_color_stream = streams['rep_color']

          print('annotations.keys()',annotations.keys())
          
          bounding_box = []
          image_points = []

          for idx in annotations.keys():

            frame_annotations = annotations[idx]

            frame_annotations.pop('quality', None)

            img_points= np.array([frame_annotations[key] for key in sorted(frame_annotations.keys())])

            print("img_points",img_points, img_points.shape)

            bounding_box.append(img_points)
            image_points.append(np.zeros((2,2)))

            sorted_keys= sorted(frame_annotations.keys())

            print('self.streams.image_points',self.streams['color'].image_points)

          # transform annotations
          video = ff.vf.transform_annotations(  directory=self.original_directory, 
                                                extension=self.original_extension, 
                                                bounding_box=bounding_box, 
                                                image_points=image_points, 
                                                source_stream=color_stream, 
                                                destination_stream=rep_color_stream, 
                                                n_frames=self.n_frames)

          for idx, image in enumerate(video.as_array()): # next line is not loading the data but just use the projection , probably wont work

          # for idx in annotations.keys():

            print('self.streams',self.streams,image.shape)

            rep_image_points = image_points[idx]

            print("rep_image_points",rep_image_points)

            if rep_image_points:

            
              rep_frame_annotations= {}

              for idx, sk in enumerate(sorted_keys):

                rep_frame_annotations[sk]=[rep_image_points[idx,:][0],rep_image_points[idx,:][1]]

              if rep_frame_annotations:

                    rep_annotations[str(idx)] = rep_frame_annotations



        return rep_annotations



# import json

# file_path= '/idiap/temp/ageorge/HQWMCA_NEW_API/WARP/UNORM/annotations/01.03.19/1_01_0001_0000_00_00_000-241b2419.json'

# with open(file_path, 'r') as json_file:

#     annotations = json.load(json_file)

# aa=annotations['0']

# aa.pop('quality', None)


# for key in sorted(aa.keys()): 
#     ...:     a.append(aa[key]) 



# [aa[key] for key in sorted(aa.keys())]

# np.array(a)

# annot_entries=sorted(aa.keys())
