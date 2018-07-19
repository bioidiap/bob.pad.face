#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guillaume Clivaz
"""
import json
import os
import bob.io.base
import tables
import pkg_resources
import collections

from pandas import read_csv
from collections import defaultdict

from bob.pad.base.database import PadDatabase, PadFile
from bob.bio.video import FrameSelector
from bob.extension import rc
from bob.pad.face.preprocessor.FaceCropAlign import detect_face_landmarks_in_image

from batl.utils.data import load_video_stream_from_hdf5
from batl.utils.data import load_data_config
from batl.utils.data import convert_arrays_to_frame_container, generate_odin_config
from batl.utils.h5adapter import h5adapter

DEFAULT_GT_CONFIG = dict(client_id=0, type_id=1, pai_id=2, low_level_group=3)

class BatlDockerPadFile(PadFile):
    """
    A high level implementation of the File class for the BATL Docker
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
            A dictionary containing client_id, type_id, path and file_id,
            replacing the File class instance as no low level db is used,
            but instead a ground-truth.csv is given as input in
            BatlDockerPadDatabase

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
        if f['type_id'] >= 1:
            attack_type = 'attack'
        else:
            attack_type = None

        super(BatlDockerPadFile, self).__init__(
            client_id=f['client_id'],
            path=f['path'],
            attack_type=attack_type,
            file_id=f['file_id'])

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
            String containing the path to BATL GOVT database.
            Default: ``None``.

        ``extension`` : :py:class:`str`
            Extension of the BATL GOVT database.
            Default: ".h5".

        ``frame_selector`` : :any:`bob.bio.video.FrameSelector`, optional
            Specifying the frames to be selected.

        **Returns:**

        ``data`` : FrameContainer
            Video data stored in the FrameContainer,
            see ``bob.bio.video.utils.FrameContainer``
            for further details.
        """
        modality=self.stream_type
        reference_stream_type=self.reference_stream_type
        warp_to_reference=self.warp_to_reference
        convert_to_rgb=self.convert_to_rgb
        crop=self.crop
        max_frames=self.max_frames

        config_path = pkg_resources.resource_filename('bob.pad.face', 'config/batl_docker_configs/databases/data_config.json')
        config = load_data_config(config_path)
        filepath = self.make_path(directory, extension)
        ret = {}
        with tables.open_file(filepath, 'r') as hdf5_file:
            configs = generate_odin_config(config, hdf5_file)
            config = next(iter(configs.values()))
            if modality == 'all':
                modalities = list(config.keys())
            elif isinstance(modality, str):
                modalities = [modality]
            elif isinstance(modality, collections.Iterable):
                modalities = list(modality)
            for mod in modalities:
                video, timestamps, masks = \
                        load_video_stream_from_hdf5(h5adapter(hdf5_file), mod, reference_stream_type,
                                                    config,
                                                    warp_to_reference=warp_to_reference,
                                                    convert_to_rgb=convert_to_rgb, crop=crop,
                                                    max_frames=max_frames)
                video = convert_arrays_to_frame_container(video)
                ret[mod] = {'video':video,  'timestamps':timestamps,  'masks':masks}
            if len(modalities) == 1:
                ret = ret[modality]
            data = ret

        for meta_data in data.keys():
            if meta_data != 'rppg':
                data[meta_data] = frame_selector(data[meta_data])

        if self.video_data_only:

            data = data['video']

        return data

class BatlDockerPadDatabase(PadDatabase):
    """
    A high level implementation of the Database class modified to use
    a ground-truth.csv file instead of a low-level database.
    """

    def __init__(
            self,
            protocol='baseline',
            original_directory="/tmp/sub_dir/data/",
            original_extension='.h5',
            annotations_temp_dir="",
            landmark_detect_method="mtcnn",
            exlude_attacks_list=None,
            ground_truth={'govt':{'path':'/tmp/sub_dir/gt.csv',
                                  'config':DEFAULT_GT_CONFIG}},
            retrain=False,
            **kwargs):
        """
        **Parameters:**

        ``protocol`` : str or None
            The name of the protocol that defines the default experimental
            setup for this database. Also a "complex" protocols can be
            parsed.
            For example:
            "baseline-color-5" - baseline protocol, color data only,
            use 5 first frames.
            "baseline-depth-5" - baseline protocol, depth data only,
            use 5 first frames.
            "baseline-color" - baseline protocol, depth data only, use all frames.
            "baseline-infrared-50-join_train_dev" - baseline protocol,
            infrared data only, use 50 frames, join train and dev sets forming
            a single large training set.
            See the ``parse_protocol`` method of this class.

        ``original_directory`` : str
            The directory where the original data of the ground-truth.csv are stored.

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

        def read_ground_truth(file_path, gt_type = None):
            gt_csv = None
            try:
                gt_csv = read_csv(file_path)
            except Exception as e:
                print(e)

            if gt_type != 'Idiap':
                gt_csv.rename(columns={'h5_file': 'path', 'face_15': 'type_id'}, inplace=True)
                gt_csv = gt_csv[['path','any_pa','type_id']]
                gt_csv['group'] = "train"
                gt_csv['pai_id'] = None

            # remove ".h5" in file_path
            gt_csv['path'] = gt_csv['path'].apply(lambda x: os.path.splitext(x)[0])
            gt_csv['client_id'] = gt_csv['path'].apply(lambda x:  x.strip().split('/')[-1])
            return gt_csv

        # Place input data from govt ground-truth in train set, given without any pai_id
        gt_dataframe = read_ground_truth(ground_truth['govt']['path'])

        # If retraining, place data from idiap ground-truth in train/eval/dev set
        if retrain:
            gt_dataframe_idiap = read_ground_truth(ground_truth['idiap']['path'],'Idiap')
            gt_dataframe = gt_dataframe.append(gt_dataframe_idiap, ignore_index=True, sort = False)

        # Sort in defaultdict with group as keys and list of dict (client_id, path,...) as values
        gt_dataframe.index += 1
        gt_dataframe['file_id'] = gt_dataframe.index
        gt_dataframe = gt_dataframe.set_index('group')[['client_id','path','type_id','pai_id','file_id']]
        gt_dataframe['group'] = gt_dataframe.index
        self.gt_list = gt_dataframe.apply(dict,1)\
                                   .groupby(level=0)\
                                   .agg(lambda x: list(x.values))\
                                   .to_dict(into=defaultdict(list))

        self.low_level_group_names = (
            'train', 'validation',
            'test')  # group names in the low-level database interface
        self.high_level_group_names = (
            'train', 'dev',
            'eval')  # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        super(BatlDockerPadDatabase, self).__init__(
            name='batldocker',
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

    #@property
    #def original_directory(self):
    #   return self.db.original_directory

    #@original_directory.setter
    #def original_directory(self, value):
    #    self.db.original_directory = value

    def parse_protocol(self, protocol):
        """
        Parse the protocol name, which is give as a string.
        An example of protocols it can parse:
        "baseline-color-5" - baseline protocol, color data only, use 5 first frames.
        "baseline-depth-5" - baseline protocol, depth data only, use 5 first frames.
        "baseline-color" - baseline protocol, depth data only, use all frames.

        **Parameters:**

        ``protocol`` : str
            Protocol name to be parsed. Example: "baseline-depth-5" .

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


    def objects(self,
                protocol=None,
                groups=None,
                purposes=None,
                model_ids=None,
                **kwargs):
        """
        This function returns lists of BatlDockerPadFile objects, which fulfill the
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

        ``files`` : [BatlDockerPadFile]
            A list of BatlDockerPadFile objects.
        """
        if protocol is None:
            protocol = self.protocol

        if groups is None:
            groups = self.high_level_group_names

        if purposes is None:
            purposes = ['real', 'attack']
        # protocol = baseline useless
        # extra useless
        protocol, stream_type, max_frames, extra = self.parse_protocol(protocol)
        

        if not isinstance(groups, list) and groups is not None:  # if a single group is given make it a list
            groups = list(groups)

        if groups is None:
            groups = self.high_level_group_names

        # filter the groups needed
        selected_list = []
        for group in groups:
            selected_list = selected_list + self.gt_list[group]

        # filter the purpose 'real' or 'attack'
        files = []
        for f in selected_list:
            if purposes == 'real':
                if f['type_id']== 0:
                    files.append(f)
            elif purposes == 'attack':
                if f['type_id'] >= 1:
                    files.append(f)
            elif purposes == ['real','attack']:
                files.append(f)
            else:
                files.append(f)

        #if groups == 'train' or 'train' in groups and len(groups) == 1:
        #    # exclude "makeup" case
        #    if self.exlude_attacks_list is not None and "makeup" in self.exlude_attacks_list:

        #        files = [f for f in files if f.pai_id != 4] # excludes makeup

        # check parameters for validity
        self.check_parameters_for_validity(protocol, "protocol", ['baseline'])
        self.check_parameters_for_validity(purposes, "purposes", ['real','attack',['real','attack']])

        files = [BatlDockerPadFile(f, stream_type, max_frames) for f in files]

        return files

    def annotations(self, f):
        """
        Computes annotations for a given file object ``f``, which
        is an instance of the ``BatlDockerPadFile`` class.

        NOTE: you can pre-compute annotation in your first experiment
        and then reuse them in other experiments setting
        ``self.annotations_temp_dir`` path of this class, where
        precomputed annotations will be saved.

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of ``BatlDockerPadFile`` defined above.

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

        file_path = os.path.join(self.annotations_temp_dir, f.f['path'] + ".json")

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
