#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#==============================================================================
from bob.pad.base.database import PadFile # Used in ReplayPadFile class

from bob.pad.base.database import PadDatabase

# Import HLDI for the databases to aggregate:
from bob.pad.face.database import replay as replay_hldi

from bob.pad.face.database import replay_mobile as replay_mobile_hldi

from bob.pad.face.database import msu_mfsd as msu_mfsd_hldi

import numpy as np

#==============================================================================
class AggregatedDbPadFile(PadFile):
    """
    A high level implementation of the File class for the Aggregated Database
    uniting 3 databases: REPLAY-ATTACK, REPLAY-MOBILE and MSU MFSD.
    """

    def __init__(self, f):
        """
        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level db interface
            of the Replay-Attack or Replay-Mobile or MSU MFSD database, respectively
            in the bob.db.replay.models.py       file or
            in the bob.db.replaymobile.models.py file or
            in the bob.db.msu_mfsd_mod.models.py file.
        """

        self.f = f
        # this f is actually an instance of the File class that is defined in
        # bob.db.<database_name>.models and the PadFile class here needs
        # client_id, path, attack_type, file_id for initialization. We have to
        # convert information here and provide them to PadFile. attack_type is a
        # little tricky to get here. Based on the documentation of PadFile:
        # In cased of a spoofed data, this parameter should indicate what kind of spoofed attack it is.
        # The default None value is interpreted that the PadFile is a genuine or real sample.
        if f.is_real():
            attack_type = None
        else:
            attack_type = 'attack'
        # attack_type is a string and I decided to make it like this for this
        # particular database. You can do whatever you want for your own database.

        file_path = self.encode_file_path(f)

        file_id = self.encode_file_id(f)

        super(AggregatedDbPadFile, self).__init__(client_id = f.client_id, path = file_path,
                                                  attack_type = attack_type, file_id = file_id)


    #==========================================================================
    def encode_file_id(self, f, n = 2000):
        """
        Return a modified version of the ``f.id`` ensuring uniqueness of the ids
        across all databases.

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level db interface
            of the Replay-Attack or Replay-Mobile or MSU MFSD database, respectively
            in the bob.db.replay.models.py       file or
            in the bob.db.replaymobile.models.py file or
            in the bob.db.msu_mfsd_mod.models.py file.

        ``n`` : :py:class:`int`
            An offset to be added to the file id for different databases is defined
            as follows: offset = k*n, where k is the database number,
            k = 0,1,2 in our case. Default: 2000.

        **Returns:**

        ``file_id`` : :py:class:`int`
            A modified version of the file id, which is now unigue accross
            all databases.
        """

        import bob.db.replay
        import bob.db.replaymobile
        import bob.db.msu_mfsd_mod

        if isinstance(f, bob.db.replay.models.File): # check if instance of File class of LLDI of Replay-Attack

            file_id = f.id

        if isinstance(f, bob.db.replaymobile.models.File): # check if instance of File class of LLDI of Replay-Mobile

            file_id = np.int(f.id + n)

        if isinstance(f, bob.db.msu_mfsd_mod.models.File): # check if instance of File class of LLDI of MSU MFSD

            file_id = np.int(f.id + 2*n)

        return file_id


    #==========================================================================
    def encode_file_path(self, f):
        """
        Append the name of the database to the end of the file path separated
        with "_".

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level db interface
            of the Replay-Attack or Replay-Mobile or MSU MFSD database, respectively
            in the bob.db.replay.models.py       file or
            in the bob.db.replaymobile.models.py file or
            in the bob.db.msu_mfsd_mod.models.py file.

        **Returns:**

        ``file_path`` : :py:class:`str`
            Modified path to the file, with database name appended to the end
            separated with "_".
        """

        import bob.db.replay
        import bob.db.replaymobile
        import bob.db.msu_mfsd_mod

        if isinstance(f, bob.db.replay.models.File): # check if instance of File class of LLDI of Replay-Attack

            file_path = '_'.join([f.path, 'replay'])

        if isinstance(f, bob.db.replaymobile.models.File): # check if instance of File class of LLDI of Replay-Mobile

            file_path = '_'.join([f.path, 'replaymobile'])

        if isinstance(f, bob.db.msu_mfsd_mod.models.File): # check if instance of File class of LLDI of MSU MFSD

            file_path = '_'.join([f.path, 'msu_mfsd_mod'])

        return file_path


    #==========================================================================
    def load(self, directory=None, extension='.mov'):
        """
        Overridden version of the load method defined in the ``PadFile``.

        **Parameters:**

        ``directory`` : :py:class:`str`
            String containing the paths to all databases used in this aggregated
            database. The paths are separated with a space.

        ``extension`` : :py:class:`str`
            Extension of the video files in the REPLAY-ATTACK and REPLAY-MOBILE
            databases. The extension of files in MSU MFSD is not taken into account
            in the HighLevel DB Interface of MSU MFSD. Default: '.mov'.

        **Returns:**

        ``video_data`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.
        """

        import bob.db.replay
        import bob.db.replaymobile
        import bob.db.msu_mfsd_mod

        directories = directory.split(" ")

        if isinstance(self.f, bob.db.replay.models.File): # check if instance of File class of LLDI of Replay-Attack

            db_pad_file = replay_hldi.ReplayPadFile(self.f) # replay_hldi is HLDI of Replay-Attack

            directory = directories[0]

        if isinstance(self.f, bob.db.replaymobile.models.File): # check if instance of File class of LLDI of Replay-Mobile

            db_pad_file = replay_mobile_hldi.ReplayMobilePadFile(self.f) # replay_mobile_hldi is HLDI of Replay-Mobile

            directory = directories[1]

        if isinstance(self.f, bob.db.msu_mfsd_mod.models.File): # check if instance of File class of LLDI of MSU MFSD

            db_pad_file = msu_mfsd_hldi.MsuMfsdPadFile(self.f) # msu_mfsd_hldi is HLDI of MSU MFSD

            directory = directories[2]

        video_data = db_pad_file.load(directory = directory, extension = extension)

        return video_data # video data


#==============================================================================
class AggregatedDbPadDatabase(PadDatabase):
    """
    A high level implementation of the Database class for the Aggregated Database
    uniting 3 databases: REPLAY-ATTACK, REPLAY-MOBILE and MSU MFSD.
    """

    def __init__(
        self,
        protocol='grandtest', # grandtest is the default protocol for this database
        original_directory=None,
        original_extension=None,
        **kwargs):
        """
        **Parameters:**

        ``protocol`` : :py:class:`str` or ``None``
            The name of the protocol that defines the default experimental setup
            for this database. Default: 'grandtest'.

        ``original_directory`` : :py:class:`str`
            String containing the paths to all databases used in this aggregated
            database. The paths are separated with a space. Default: None.

        ``original_extension`` : :py:class:`str`
            Extension of the video files in the REPLAY-ATTACK and REPLAY-MOBILE
            databases. The extension of files in MSU MFSD is not taken into account
            in the HighLevel DB Interface of MSU MFSD. Default: None.

        ``kwargs``
            The arguments of the :py:class:`bob.bio.base.database.BioDatabase` base class constructor.
        """

        # Import LLDI for all databases:
        import bob.db.replay
        import bob.db.replaymobile
        import bob.db.msu_mfsd_mod

        self.replay_db = bob.db.replay.Database()
        self.replaymobile_db = bob.db.replaymobile.Database()
        self.msu_mfsd_db = bob.db.msu_mfsd_mod.Database()

        # Since the high level API expects different group names than what the low
        # level API offers, you need to convert them when necessary
        self.low_level_group_names = ('train', 'devel', 'test') # group names in the low-level database interface
        self.high_level_group_names = ('train', 'dev', 'eval') # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        super(AggregatedDbPadDatabase, self).__init__(
            name = 'aggregated_db',
            protocol = protocol,
            original_directory = original_directory,
            original_extension = original_extension,
            **kwargs)


    #==========================================================================
    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        """
        This function returns a list of AggregatedDbPadFile objects, which fulfill the given restrictions.

        Keyword parameters:

        ``groups`` : :py:class:`str`
            OR a list of strings.
            The groups of which the clients should be returned.
            Usually, groups are one or more elements of ('train', 'dev', 'eval')

        ``protocol`` : :py:class:`str`
            The protocol for which the clients should be retrieved.
            The protocol is dependent on your database.
            If you do not have protocols defined, just ignore this field.

        ``purposes`` : :py:class:`str`
            OR a list of strings.
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.

        ``model_ids``
            This parameter is not supported in PAD databases yet

        **Returns:**

        ``files`` : [AggregatedDbPadFile]
            A list of AggregatedDbPadFile objects.
        """

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.
        replay_files = self.replay_db.objects(protocol=protocol, groups=groups, cls=purposes, **kwargs)

        replaymobile_files = self.replaymobile_db.objects(protocol=protocol, groups=groups, cls=purposes, **kwargs)

        msu_mfsd_files = self.msu_mfsd_db.objects(group=groups, cls=purposes, **kwargs)

        files = replay_files + replaymobile_files + msu_mfsd_files # append all files to a single list

        files = [AggregatedDbPadFile(f) for f in files]
        return files


    #==========================================================================
    def annotations(self, f):
        """
        Return annotations for a given file object ``f``, which is an instance
        of ``AggregatedDbPadFile`` defined in the HLDI of the Aggregated DB.
        The ``load()`` method of ``AggregatedDbPadFile`` class (see above)
        returns a video, therefore this method returns bounding-box annotations
        for each video frame. The annotations are returned as dictionary of dictionaries.

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of ``AggregatedDbPadFile`` defined above.

        **Returns:**

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box in frame N.
        """

        import bob.db.replay
        import bob.db.replaymobile
        import bob.db.msu_mfsd_mod

        directories = self.original_directory.split(" ")

        if isinstance(f.f, bob.db.replay.models.File): # check if instance of File class of LLDI of Replay-Attack

            hldi_db = replay_hldi.ReplayPadDatabase(original_directory = directories[0])

        if isinstance(f.f, bob.db.replaymobile.models.File): # check if instance of File class of LLDI of Replay-Mobile

            hldi_db = replay_mobile_hldi.ReplayMobilePadDatabase(original_directory = directories[1])

        if isinstance(f.f, bob.db.msu_mfsd_mod.models.File): # check if instance of File class of LLDI of MSU MFSD

            hldi_db = msu_mfsd_hldi.MsuMfsdPadDatabase(original_directory = directories[2])

        annotations = hldi_db.annotations(f)

        return annotations














