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
            exlude_attacks_list=None,
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
            "nowig-infrared-50-join_train_dev" - nowig protocol,
            infrared data only, use 50 frames, join train and dev sets forming
            a single large training set.
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

        ``exlude_attacks_list`` : [str]
            A list of strings defining which attacks should be excluded from
            the training set. This shoould be handled in ``objects()`` method.
            Currently handled attacks: "makeup".
            Default: ``None``.

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
        self.exlude_attacks_list = exlude_attacks_list

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

        ``extra`` : str
            An extra string which is handled in ``self.objects()`` method.
            Extra strings which are currently handled are defined in
            ``possible_extras`` of this function.
            For example, if ``extra="join_train_dev"``, the train and dev
            sets will be joined in ``self.objects()``,
            forming a single training set.
        """

        possible_extras = ['join_train_dev']

        components = protocol.split("-")

        extra = [item for item in possible_extras if item in components]

        extra = extra[0] if extra else None

        if extra is not None:
            components.remove(extra)

        components += [None, None]

        components = components[0:3]

        protocol, stream_type, max_frames = components

        if max_frames is not None:

            max_frames = int(max_frames)

        return protocol, stream_type, max_frames, extra

    def _fix_funny_eyes_in_objects(self, protocol, groups, purposes):
        """
        This function redistributes FunnyEyes PAs accross 'train', 'dev' and
        'eval' sets in the following way.

        Original (low-level DB) distribution is as follows:
        'train' = N1
        'dev' = N2
        'eval' = N3

        After this function is applied the distribution is:
        'train' = N1 + 1/2*N2
        'dev' = N2 - 1/2*N2
        'eval' = N3

        **Parameters:**

        ``protocol`` : str
            The protocol for which the clients should be retrieved.

        ``groups`` : :py:class:`str`
            OR a list of strings.
            The groups of which the clients should be returned.
            Usually, groups are one or more elements of ('train', 'dev', 'eval')

        ``purposes`` : :obj:`str` or [:obj:`str`]
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.

        **Returns:**

        ``files`` : [VideoFile]
            A list of VideoFile objects defined in BATL Low Level Database
            Interface.
        """

        if groups is None:
            groups = self.low_level_group_names

        files_train = []
        files_dev = []
        files_eval = []

        if groups == 'train' or 'train' in groups:

            files_train = self.db.objects(protocol=protocol, groups='train', purposes=purposes)

            files_to_append = self.db.objects(protocol=protocol, groups='validation', purposes=purposes)

            exclude = ["_1_01", "_1_04", "_1_05", "_1_06", "_1_07"] # files ending with these paths relate to FunnyEyes

            files_to_append = [f for f in files_to_append if f.path[-5:] in exclude]

            files_to_append = files_to_append[0:int(len(files_to_append)/2)] # append HALF of files from "dev" to "train" set

            files_train = files_train + files_to_append

        if groups == 'validation' or 'validation' in groups:

            files_dev = self.db.objects(protocol=protocol, groups='validation', purposes=purposes)

            exclude = ["_1_01", "_1_04", "_1_05", "_1_06", "_1_07"] # files ending with these paths relate to FunnyEyes

            files_to_append_1 = [f for f in files_dev if f.path[-5:] in exclude] # "dev" files containing FunnyEyes

            files_to_append_1 = files_to_append_1[-int(len(files_to_append_1)/2):] # second HALF of "dev" files containing FunnyEyes

            files_to_append_2 = [f for f in files_dev if f.path[-5:] not in exclude] # "dev" set without FunnyEyes

            files_dev = files_to_append_1 + files_to_append_2

        if groups == 'test' or 'test' in groups:

            files_eval = self.db.objects(protocol=protocol, groups='test', purposes=purposes) # this group remain unchanged

        files = files_train + files_dev + files_eval

        return files

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

        ``groups`` : :py:class:`str`
            OR a list of strings.
            The groups of which the clients should be returned.
            Usually, groups are one or more elements of ('train', 'dev', 'eval')

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

        protocol, stream_type, max_frames, extra = self.parse_protocol(protocol)

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(
            groups, self.low_level_group_names, self.high_level_group_names)

        if not isinstance(groups, list) and groups is not None:  # if a single group is given make it a list
            groups = list(groups)

        if extra is not None and "join_train_dev" in extra:

            if groups == ['train']: # join "train" and "dev" sets
                files = self.db.objects(protocol=protocol,
                                        groups=['train', 'validation'],
                                        purposes=purposes, **kwargs)

            # return ALL data if "train" and "some other" set/sets are requested
            elif len(groups)>=2 and 'train' in groups:
                files = self.db.objects(protocol=protocol,
                                        groups=self.low_level_group_names,
                                        purposes=purposes, **kwargs)

            # addresses the cases when groups=['validation'] or ['test'] or ['validation', 'test']:
            else:
                files = self.db.objects(protocol=protocol,
                                        groups=['test'],
                                        purposes=purposes, **kwargs)

        else:
            files = self._fix_funny_eyes_in_objects(protocol=protocol,
                                                    groups=groups,
                                                    purposes=purposes, **kwargs)

        if groups == 'train' or 'train' in groups and len(groups) == 1:
            # exclude "makeup" case
            if self.exlude_attacks_list is not None and "makeup" in self.exlude_attacks_list:

                files = [f for f in files if os.path.split(f.path)[-1].split("_")[-2:-1][0] != "5"]

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
