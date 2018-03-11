#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Used in BATLMobilePadFile class
from bob.pad.base.database import PadDatabase, PadFile
from bob.bio.video import FrameSelector
from bob.extension import rc
from bob.db.batl.batl_config import BATL_CONFIG

import pkg_resources
from batl.utils.data import load_data_config

from bob.pad.face.preprocessor.FaceCropAlign import detect_face_landmarks_in_image

import json

class BatlPadFile(PadFile):
    """
    A high level implementation of the File class for the BATL
    database.
    """

    def __init__(self, f,
                 stream_type,  # a list of streams to be loaded
                 max_frames,
                 reference_stream_type="color",
                 warp_to_reference=True,
                 convert_to_rgb=False,
                 crop=None):

        """
        Parameters
        ----------
        f : object
            An instance of the File class defined in the low level db interface
            of the BATL database, in the bob.db.batl.models.py file.
        """

        self.f = f
        if f.is_attack():
            attack = batl_config[f.type_id]
            attack_type = '{} : {}'.format(attack['name'], attack['pai'][f.pai_id])
        else:
            attack_type = None

        super(BatlPadFile, self).__init__(
            client_id=f.client_id,
            path=f.path,
            attack_type=attack_type,
            file_id=f.id)

        self.stream_type = stream_type
        self.max_frames = max_frames
        self.reference_stream_type = reference_stream_type  # "color"
        self.data_format_config = load_data_config(pkg_resources.resource_filename('batl.utils', 'config/idiap_hdf5_data_config.json'))
        self.warp_to_reference = warp_to_reference  # True
        self.convert_to_rgb = convert_to_rgb  # False
        self.crop = crop  # None


    def load(self, directory=None, extension='.hdf5', frame_selector=FrameSelector(selection_style='all')):

        data = f.load(self, directory=directory,
                      extension=extension,
                      modality=self.stream_type, # TODO: this parameter is currently missing in bob.db.batl, add it there
                      reference_stream_type=self.reference_stream_type,
                      data_format_config=self.data_format_config,
                      warp_to_reference=self.warp_to_reference,
                      convert_to_rgb=self.convert_to_rgb,
                      crop=self.crop,
                      max_frames=self.max_frames)

        for modality in data.keys():
            if modality != 'rppg':
                data[modality] = frame_selector(data[modality])
        return data


class BatlPadDatabase(PadDatabase):
    """
    A high level implementation of the Database class for the BATL
    database.
    """

    def __init__(
            self,
            # grandtest is the default protocol for this database
            protocol='grandtest',
            original_directory=rc['bob.db.batl.directory'],
            original_extension='.h5',
            annotations_temp_dir="",
            landmark_detect_method="mtcnn",
            **kwargs):
        """
        Parameters
        ----------

        protocol : str or None
            The name of the protocol that defines the default experimental
            setup for this database. Also a "complex" protocols can be
            parsed.
            For example:
            "grandtest-color-5" - grandtest protocol, color data only, use 5 first frames.
            "grandtest-depth-5" - grandtest protocol, depth data only, use 5 first frames.
            "grandtest-color" - grandtest protocol, depth data only, use all frames.
            See the ``parse_protocol`` method of this class.

        original_directory : str
            The directory where the original data of the database are stored.

        original_extension : str
            The file name extension of the original data.

        kwargs
            The arguments of the :py:class:`bob.bio.base.database.BioDatabase`
            base class constructor.
        """

        from bob.db.batl import Database as LowLevelDatabase

        self.db = LowLevelDatabase()

        # Since the high level API expects different group names than what the
        # low level API offers, you need to convert them when necessary
        self.low_level_group_names = (
            'train', 'validation',
            'test')  # group names in the low-level database interface
        self.high_level_group_names = (
            'train', 'dev',
            'eval')  # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        super(BatlPadDatabase, self).__init__(
            name='batl',
            protocol=protocol,
            original_directory=original_directory,
            original_extension=original_extension,
            **kwargs)

        self.protocol = protocol
        self.original_directory = original_directory
        self.original_extension = original_extension
        self.annotations_temp_dir = annotations_temp_dir
        self.landmark_detect_method = landmark_detect_method

    @property
    def original_directory(self):
        return self.db.original_directory

    @original_directory.setter
    def original_directory(self, value):
        self.db.original_directory = value

    def parse_protocol(self, protocol):
        """
        Parse the protocol name, which is give as a string.
        An example of protocols it can parse:
        "grandtest-color-5" - grandtest protocol, color data only, use 5 first frames.
        "grandtest-depth-5" - grandtest protocol, depth data only, use 5 first frames.
        "grandtest-color" - grandtest protocol, depth data only, use all frames.

        **Parameters:**

        ``protocol`` : str
            Protocol name to be parsed. Example: "grandtest-depth-5" .

        **Returns:**

        ``protocol`` : str
            The name of the protocol as defined in the low level db interface.

        ``stream_types`` : str
            The name of the channel/stream_type to be loaded.

        ``max_frames`` : int
            The number of frames to be loaded.
        """

        components = protocol.split("-")

        components = components + [None, None]

        components = components[0:3]

        protocol, stream_types, max_frames = components

        if max_frames is not None:

            max_frames = int(max_frames)

        return protocol, stream_types, max_frames

    def objects(self,
                protocol=None,
                groups=None,
                purposes=None,
                sessions=None,
                **kwargs):
        """
        This function returns lists of BATLPadFile objects, which fulfill the
        given restrictions.

        Parameters
        ----------
        protocol : str
            The protocol for which the clients should be retrieved.
            The protocol is dependent on your database.
            If you do not have protocols defined, just ignore this field.

        purposes : :obj:`str` or [:obj:`str`]
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.

        model_ids
            This parameter is not supported in PAD databases yet
        **kwargs

        Returns
        -------
        files : [BATLPadFile]
            A list of BATLPadFile objects.
        """

        protocol, stream_types, max_frames = self.parse_protocol(protocol)

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(
            groups, self.low_level_group_names, self.high_level_group_names)
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.
        files = self.db.objects(protocol=protocol, groups=groups, purposes=purposes **kwargs)


#        files = self.db.objects(protocol=protocol, purposes=groups, **kwargs)
#
#        if purposes == ["real", "attack"]:
#
#            files = files
#
#        if purposes == "real" or purposes == ["real"]:
#
#            a = 1
#
#
#        if purposes == "attack" or purposes == ["attack"]:
#
#            a = 1




        files = [BatlPadFile(f, stream_type, max_frames) for f in files]
        return files

    def annotations(self, f):

        file_path = os.path.join(self.annotations_temp_dir, f.f.path + ".json")

        if not os.path.isfile(file_path): # no file with annotations

            video = f.f.load(self, directory=self.original_directory,
                             extension=self.original_extension,
                             modality="color", # TODO: this parameter is currently missing in bob.db.batl, add it there
                             reference_stream_type="color",
                             data_format_config=load_data_config(pkg_resources.resource_filename('batl.utils', 'config/idiap_hdf5_data_config.json')),
                             warp_to_reference=False,
                             convert_to_rgb=False,
                             crop=None,
                             max_frames=None)

            annotations = {}

            for idx, image in enumerate(video):

                frame_annotations = detect_face_landmarks_in_image(image, method = self.landmark_detect_method)

                if frame_annotations:

                    annotations[str(idx)] = frame_annotations

            if self.annotations_temp_dir: # if directory is not an empty string

                with open(file_path, 'w') as json_file:

                    json_file.write(json.dumps(annotations))

        else: # if file with annotations exists load them from file

            with open(file_path, 'r') as json_file:

                annotations = json.load(json_file)

        return annotations



