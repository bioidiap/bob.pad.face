#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import bob.io.base
from bob.pad.base.database import PadDatabase, PadFile
from bob.extension import rc
from bob.pad.face.preprocessor.FaceCropAlign import detect_face_landmarks_in_image

from bob.db.hqwmca.attack_dictionaries import idiap_type_id_config, idiap_subtype_id_config
from bob.io.stream import stream


# def _color(f):
#   return f.stream('color')


dark_for_stereo     = False
subtract_dark_nir   = False
subtract_dark_swir  = True 

color = stream('color')

nir_735nm       = stream('nir_left_735nm').adjust(color)
nir_850nm       = stream('nir_left_850nm').adjust(color)
nir_940nm       = stream('nir_left_940nm').adjust(color)
nir_1050nm      = stream('nir_left_1050nm').adjust(color)

nir_left_dark   = stream('nir_left').adjust(color)
nir_right_dark  = stream('nir_left').adjust(color)
nir_left_stereo = stream('nir_left_stereo').adjust(color)
nir_right_stereo= stream('nir_right_stereo').adjust(color)

if subtract_dark_nir:
    nir_735nm   = nir_735nm.subtract_dark(nir_left_dark)
    nir_850nm   = nir_850nm.subtract_dark(nir_left_dark)
    nir_940nm   = nir_940nm.subtract_dark(nir_left_dark)
    nir_1050nm  = nir_1050nm.subtract_dark(nir_left_dark)

nir = nir_735nm.stack(nir_850nm).stack(nir_940nm).stack(nir_1050nm)

if dark_for_stereo:
    nir_left_stereo   = nir_left_dark
    nir_right_stereo  = nir_right_dark


from bob.ip.stereo import StereoParameters

stereo_parameters = StereoParameters()

stereo_parameters.erode   = True  # remove small features
stereo_parameters.inpaint   = True  # fill holes
_map_3d      = nir_left_stereo.stereo(nir_right_stereo, stereo_parameters=stereo_parameters)


# def dd(a):
#   return a.select(channel=2) 


# depth       = dd(_map_3d)

depth = _map_3d.select(channel=2).normalize(tmin=20.0, tmax=100.0, dtype='uint16') 

color       = color.reproject(nir_left_stereo, nir_right_stereo, _map_3d)



streams = { 'color'     : color}


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

    def __init__(self, vf, stream_file=None, streams=None, n_frames=10):
      """ Init

      Parameters
      ----------
      vf : :py:class:`object`
        An instance of the VideoFile class defined in the low level db interface
        of the HQWMCA database, in the bob.db.hqwmca.models.py file.
      stream_file: 
        bob.io.stream StreamFile object from which the stream are derievd from.
      streams: :py:dict:
        Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file
      n_frames: int:
        The number of frames, evenly spread, you would like to retrieve 
      
      """
      self.vf = vf
      self.stream_file = stream_file
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
        return self.vf.load(directory, extension, stream_file=self.stream_file, streams=self.streams, n_frames=self.n_frames)


class HQWMCAPadDatabase(PadDatabase): 
    """High level implementation of the Database class for the HQWMCA database.
   
    Attributes
    ----------
    db : :py:class:`bob.db.hqwmca.Database`
      the low-level database interface
    stream_file: 
      bob.io.stream StreamFile object from which the stream are derievd from.
    streams: :py:dict:
      Dictionary of bob.io.stream Stream objects. Should be defined in a configuration file

    """
       
    def __init__(self, protocol='grand_test', original_directory=rc['bob.db.hqwmca.directory'], 
                 original_extension='.h5', annotations_dir=None, stream_file=None, streams=None, n_frames=10, use_curated_file_list=False, **kwargs):
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
        stream_file: 
          bob.io.stream StreamFile object from which the stream are derievd from.
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
      self.stream_file = stream_file
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




        return [HQWMCAPadFile(f, self.stream_file, self.streams, self.n_frames) for f in files]


    def annotations(self, f):
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

        file_path = os.path.join(self.annotations_dir, f.path + ".json")

        if not os.path.isfile(file_path):  # no file with annotations

            # original values of the arguments of f:


            video = f.load(directory=self.original_directory,
                           extension=self.original_extension)

            video = f.vf.load(directory=self.original_directory, extension=self.original_extension, streams=streams, n_frames=self.n_frames)['color']

            annotations = {}

            for idx, image in enumerate(video.as_array()):

                frame_annotations = detect_face_landmarks_in_image(image, method='mtcnn')


                print('frame_annotations',frame_annotations)

                if frame_annotations:
                  for key in frame_annotations.keys():
                    

                    if key!='quality':
                      frame_annotations[key]=(int(frame_annotations[key][0]),int(frame_annotations[key][1]))
                    else:
                      frame_annotations[key]=int(frame_annotations[key])


                  print('frame_annotations AFTER',frame_annotations)
                  if frame_annotations:

                      annotations[str(idx)] = frame_annotations

            if self.annotations_dir:  # if directory is not an empty string

                bob.io.base.create_directories_safe(directory=os.path.split(file_path)[0], dryrun=False)

                with open(file_path, 'w+') as json_file:

                    json_file.write(json.dumps(annotations))

        else:  # if file with annotations exists load them from file

            with open(file_path, 'r') as json_file:

                annotations = json.load(json_file)

        if not annotations:  # if dictionary is empty

            return None

        return annotations

