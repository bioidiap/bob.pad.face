#!/usr/bin/env python2
# -*- coding: utf-8 -*-


#==============================================================================
import bob.bio.video # Used in ReplayMobilePadFile class

from bob.pad.base.database import PadFile # Used in ReplayMobilePadFile class

from bob.pad.base.database import PadDatabase


#==============================================================================
class ReplayMobilePadFile(PadFile):
    """
    A high level implementation of the File class for the Replay-Mobile database.
    """

    def __init__(self, f):
        """
        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level db interface
            of the Replay-Mobile database, in the bob.db.replaymobile.models.py file.
        """

        self.f = f
        # this f is actually an instance of the File class that is defined in
        # bob.db.replaymobile.models and the PadFile class here needs
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

        super(ReplayMobilePadFile, self).__init__(client_id=f.client_id, path=f.path,
                                            attack_type=attack_type, file_id=f.id)


    #==========================================================================
    def convert_arr_to_frame_cont(self, data):
        """
        This function converts an input 4D array with frames into FrameContainer,
        where each frame is an RGB image. The dimensionality of the input array
        is [N_frames, 3, N_rows, N_cols].

        **Parameters:**

        ``data`` : 4D :py:class:`numpy.ndarray`
            An input 4D array with frames of the dimensionality:
            [N_frames, 3, N_rows, N_cols].

        **Returns:**

        ``frames`` : FrameContainer
            Resulting FrameContainer containing RGB frames.
        """

        frames = bob.bio.video.FrameContainer() # initialize the FrameContainer

        for idx, sample in enumerate(data):

            frames.add(idx, sample)

        return frames


    #==========================================================================
    def load(self, directory=None, extension='.mov'):
        """
        Overridden version of the load method defined in the ``PadFile``.

        **Parameters:**

        ``directory`` : :py:class:`str`
            String containing the path to the Replay-Mobile database.

        ``extension`` : :py:class:`str`
            Extension of the video files in the Replay-Mobile database.

        **Returns:**

        ``video_data`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.
        """

        video_data_array = self.f.load(directory = directory,
                                       extension = extension)

        video_data = self.convert_arr_to_frame_cont(video_data_array) # the result is now a FrameContainer

        return video_data


#==============================================================================
class ReplayMobilePadDatabase(PadDatabase):
    """
    A high level implementation of the Database class for the Replay-Mobile database.
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
            The name of the protocol that defines the default experimental setup for this database.

        ``original_directory`` : :py:class:`str`
            The directory where the original data of the database are stored.

        ``original_extension`` : :py:class:`str`
            The file name extension of the original data.

        ``kwargs``
            The arguments of the :py:class:`bob.bio.base.database.BioDatabase` base class constructor.
        """

        from bob.db.replaymobile import Database as LowLevelDatabase

        self.db = LowLevelDatabase()

        # Since the high level API expects different group names than what the low
        # level API offers, you need to convert them when necessary
        self.low_level_group_names = ('train', 'devel', 'test') # group names in the low-level database interface
        self.high_level_group_names = ('train', 'dev', 'eval') # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        super(ReplayMobilePadDatabase, self).__init__(
            name = 'replay-mobile',
            protocol = protocol,
            original_directory = original_directory,
            original_extension = original_extension,
            **kwargs)


    #==========================================================================
    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        """
        This function returns lists of ReplayMobilePadFile objects, which fulfill the given restrictions.

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

        ``files`` : :py:class:`str`
            A list of ReplayMobilePadFile objects.
        """

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.
        files = self.db.objects(protocol=protocol, groups=groups, cls=purposes, **kwargs)

        files = [ReplayMobilePadFile(f) for f in files]

        return files


    #==========================================================================
    def annotations(self, f):
        """
        Return annotations for a given file object ``f``, which is an instance
        of ``ReplayMobilePadFile`` defined in the HLDI of the Replay-Mobile DB.
        The ``load()`` method of ``ReplayMobilePadFile`` class (see above)
        returns a video, therefore this method returns bounding-box annotations
        for each video frame. The annotations are returned as dictionary of dictionaries.

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of ``ReplayMobilePadFile`` defined above.

        **Returns:**

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box in frame N.
        """

        annots = f.f.bbx(directory=self.original_directory) # numpy array containing the face bounding box data for each video frame, returned data format described in the f.bbx() method of the low level interface

        annotations = {} # dictionary to return

        for fn, frame_annots in enumerate(annots):

            topleft = (frame_annots[1], frame_annots[0])
            bottomright = (frame_annots[1] + frame_annots[3], frame_annots[0] + frame_annots[2])

            annotations[str(fn)] = {'topleft': topleft, 'bottomright': bottomright}

        return annotations

