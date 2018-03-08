#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Used in BATLMobilePadFile class
from bob.pad.base.database import PadDatabase, PadFile
from bob.bio.video import FrameSelector
from bob.extension import rc
from bob.db.batl.batl_config import BATL_CONFIG

import pkg_resources
from batl.utils.data import load_data_config

class BATLPadFile(PadFile):
    """
    A high level implementation of the File class for the BATL
    database.
    """

    def __init__(self, f,
                 stream_types,  # a list of streams to be loaded
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

        super(BATLPadFile, self).__init__(
            client_id=f.client_id,
            path=f.path,
            attack_type=attack_type,
            file_id=f.id)

        self.stream_types = stream_types
        self.max_frames = max_frames
        self.reference_stream_type = reference_stream_type  # "color"
        self.data_format_config = load_data_config(pkg_resources.resource_filename('batl.utils', 'config/idiap_hdf5_data_config.json'))
        self.warp_to_reference = warp_to_reference  # True
        self.convert_to_rgb = convert_to_rgb  # False
        self.crop = crop  # None


    def load(self, directory=None, extension='.hdf5', frame_selector=FrameSelector(selection_style='all')):

        data = f.load(self, directory=directory,
                      extension=extension,
                      stream_types=self.stream_types, # TODO: this parameter is currently missing in bob.db.batl, add it there
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


class BATLPadDatabase(PadDatabase):
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
            **kwargs):
        """
        Parameters
        ----------

        protocol : str or None
            The name of the protocol that defines the default experimental
            setup for this database.

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
        super(BATLPadDatabase, self).__init__(
            name='batl',
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
                protocol=None,
                group=None,
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

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(
            groups, self.low_level_group_names, self.high_level_group_names)
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.
        files = self.db.objects(protocol=protocol, groups=groups, purposes=purposes **kwargs)
        files = [BATLPadFile(f) for f in files]
        return files

    def annotations(self, f):
        pass
