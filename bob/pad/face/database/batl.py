#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Used in BATLMobilePadFile class
from bob.pad.base.database import PadDatabase, PadFile
from bob.bio.video import FrameSelector
from bob.extension import rc

from bob.pad.face.preprocessor.FaceCropAlign import detect_face_landmarks_in_image

import json

import os

import bob.io.base


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
                 crop=None,
                 video_data_only=True):

        """
        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level db interface
            of the BATL database, in the ``bob.db.batl.models.py`` file.

        ``stream_type`` : [] or :py:class:`str`
            A types of the streams to be loaded.

        ``max_frames`` : :py:class:`int`
            A maximum number of frames to be loaded. Frames are
            selected uniformly.

        ``reference_stream_type`` : :py:class:`str`
            Align/register all channels to this one.
            Default: "color".

        ``warp_to_reference`` : :py:class:`bool`
            Align/register downloaded video to ``reference_stream_type``,
            if set to ``True``.
            Default: ``True``.

        ``convert_to_rgb`` : :py:class:`bool`
            Type cast the non-RGB data to RGB data type,
            if set to ``True``.
            Default: ``False``.

        ``crop`` : []
            Pre-crop the frames if given, see ``bob.db.batl`` for more
            details.
            Default: ``None``.

        ``video_data_only`` : :py:class:`bool`
            Load video data only if ``True``. Otherwise more meta-data
            is loaded, for example timestamps for each frame.
            See the ``load()`` method in the low-level database
            interface for more details.
            Default: ``True``.
        """

        self.f = f
        if f.is_attack():
            attack_type = 'attack'
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
        self.warp_to_reference = warp_to_reference  # True
        self.convert_to_rgb = convert_to_rgb  # False
        self.crop = crop  # None
        self.video_data_only = video_data_only  # True

    def load(self, directory=None, extension='.h5',
             frame_selector=FrameSelector(selection_style='all')):
        """
        Load method of the file class.

        **Parameters:**

        ``directory`` : :py:class:`str`
            String containing the path to BATL database.
            Default: ``None``.

        ``extension`` : :py:class:`str`
            Extension of the BATL database.
            Default: ".h5".

        ``frame_selector`` : :any:`bob.bio.video.FrameSelector`, optional
            Specifying the frames to be selected.

        **Returns:**

        ``data`` : FrameContainer
            Video data stored in the FrameContainer,
            see ``bob.bio.video.utils.FrameContainer``
            for further details.
        """

        data = self.f.load(directory=directory,
                           extension=extension,
                           modality=self.stream_type,
                           reference_stream_type=self.reference_stream_type,
                           warp_to_reference=self.warp_to_reference,
                           convert_to_rgb=self.convert_to_rgb,
                           crop=self.crop,
                           max_frames=self.max_frames)

        for meta_data in data.keys():
            if meta_data != 'rppg':
                data[meta_data] = frame_selector(data[meta_data])

        if self.video_data_only:

            data = data['video']

        return data


class BatlPadDatabase(PadDatabase):
    """
    A high level implementation of the Database class for the BATL
    database.
    """

    def __init__(
            self,
            protocol='nowig',
            original_directory=rc['bob.db.batl.directory'],
            original_extension='.h5',
            annotations_temp_dir="",
            landmark_detect_method="mtcnn",
            **kwargs):
        """
        **Parameters:**

        ``protocol`` : str or None
            The name of the protocol that defines the default experimental
            setup for this database. Also a "complex" protocols can be
            parsed.
            For example:
            "nowig-color-5" - nowig protocol, color data only,
            use 5 first frames.
            "nowig-depth-5" - nowig protocol, depth data only,
            use 5 first frames.
            "nowig-color" - nowig protocol, depth data only, use all frames.
            See the ``parse_protocol`` method of this class.

        ``original_directory`` : str
            The directory where the original data of the database are stored.

        ``original_extension`` : str
            The file name extension of the original data.

        ``annotations_temp_dir`` : str
            Annotations computed in ``self.annotations(f)`` method of this
            class will be save to this directory if path is specified /
            non-empty string.
            Default: ``""``.

        ``landmark_detect_method`` : str
            Method to be used to compute annotations - face bounding box and
            landmarks. Possible options: "dlib" or "mtcnn".
            Default: ``"mtcnn"``.

        ``kwargs`` : dict
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
        "nowig-color-5" - nowig protocol, color data only, use 5 first frames.
        "nowig-depth-5" - nowig protocol, depth data only, use 5 first frames.
        "nowig-color" - nowig protocol, depth data only, use all frames.

        **Parameters:**

        ``protocol`` : str
            Protocol name to be parsed. Example: "nowig-depth-5" .

        **Returns:**

        ``protocol`` : str
            The name of the protocol as defined in the low level db interface.

        ``stream_type`` : str
            The name of the channel/stream_type to be loaded.

        ``max_frames`` : int
            The number of frames to be loaded.
        """

        components = protocol.split("-")

        components = components + [None, None]

        components = components[0:3]

        protocol, stream_type, max_frames = components

        if max_frames is not None:

            max_frames = int(max_frames)

        return protocol, stream_type, max_frames

    def objects(self,
                protocol=None,
                groups=None,
                purposes=None,
                model_ids=None,
                **kwargs):
        """
        This function returns lists of BatlPadFile objects, which fulfill the
        given restrictions.

        **Parameters:**

        ``protocol`` : str
            The protocol for which the clients should be retrieved.
            The protocol is dependent on your database.
            If you do not have protocols defined, just ignore this field.

        ``purposes`` : :obj:`str` or [:obj:`str`]
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.

        ``model_ids``
            This parameter is not supported in PAD databases yet

        **Returns:**

        ``files`` : [BatlPadFile]
            A list of BATLPadFile objects.
        """

        if protocol is None:
            protocol = self.protocol

        if groups is None:
            groups = self.high_level_group_names

        if purposes is None:
            purposes = ['real', 'attack']

        protocol, stream_type, max_frames = self.parse_protocol(protocol)

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(
            groups, self.low_level_group_names, self.high_level_group_names)
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.

        files = self.db.objects(protocol=protocol,
                                groups=groups,
                                purposes=purposes, **kwargs)

        files = [BatlPadFile(f, stream_type, max_frames) for f in files]
        return files

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

        file_path = os.path.join(self.annotations_temp_dir, f.f.path + ".json")

        if not os.path.isfile(file_path):  # no file with annotations

            f.stream_type = "color"
            f.reference_stream_type = "color"
            f.warp_to_reference = False
            f.convert_to_rgb = False
            f.crop = None
            f.video_data_only = True

            video = f.load(directory=self.original_directory,
                           extension=self.original_extension)

            annotations = {}

            for idx, image in enumerate(video.as_array()):

                frame_annotations = detect_face_landmarks_in_image(image, method=self.landmark_detect_method)

                if frame_annotations:

                    annotations[str(idx)] = frame_annotations

            if self.annotations_temp_dir:  # if directory is not an empty string

                bob.io.base.create_directories_safe(directory=os.path.split(file_path)[0], dryrun=False)

                with open(file_path, 'w+') as json_file:

                    json_file.write(json.dumps(annotations))

        else:  # if file with annotations exists load them from file

            with open(file_path, 'r') as json_file:

                annotations = json.load(json_file)

        if not annotations:  # if dictionary is empty

            return None

        return annotations
