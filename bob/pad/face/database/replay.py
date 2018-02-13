#!/usr/bin/env python2
# -*- coding: utf-8 -*-


#==============================================================================

import bob.bio.video # Used in ReplayPadFile class

from bob.pad.base.database import PadFile # Used in ReplayPadFile class

from bob.pad.base.database import PadDatabase


#==============================================================================

class ReplayPadFile(PadFile):
    """
    A high level implementation of the File class for the REPLAY-ATTACK database.
    """

    def __init__(self, f):
        """
        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level db interface
            of the Replay database, in the bob.db.replay.models.py file.
        """

        self.f = f
        # this f is actually an instance of the File class that is defined in
        # bob.db.replay.models and the PadFile class here needs
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

        super(ReplayPadFile, self).__init__(client_id=f.client_id, path=f.path,
                                            attack_type=attack_type, file_id=f.id)


    #==========================================================================
    def load(self, directory=None, extension='.mov'):
        """
        Overridden version of the load method defined in the ``PadFile``.

        **Parameters:**

        ``directory`` : :py:class:`str`
            String containing the path to the Replay database.

        ``extension`` : :py:class:`str`
            Extension of the video files in the Replay database.

        **Returns:**

        ``video_data`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.
        """

        path = self.f.make_path(directory=directory, extension=extension) # path to the video file

        frame_selector = bob.bio.video.FrameSelector(selection_style = 'all') # this frame_selector will select all frames from the video file

        video_data = frame_selector(path) # video data

        return video_data # video data


#==============================================================================
class ReplayPadDatabase(PadDatabase):
    """
    A high level implementation of the Database class for the REPLAY-ATTACK database.
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

        from bob.db.replay import Database as LowLevelDatabase

        self.db = LowLevelDatabase()

        # Since the high level API expects different group names than what the low
        # level API offers, you need to convert them when necessary
        self.low_level_group_names = ('train', 'devel', 'test') # group names in the low-level database interface
        self.high_level_group_names = ('train', 'dev', 'eval') # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        super(ReplayPadDatabase, self).__init__(
            name = 'replay',
            protocol = protocol,
            original_directory = original_directory,
            original_extension = original_extension,
            **kwargs)


    #==========================================================================
    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        """
        This function returns lists of ReplayPadFile objects, which fulfill the given restrictions.

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

        ``files`` : [ReplayPadFile]
            A list of ReplayPadFile objects.
        """

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(groups, self.low_level_group_names, self.high_level_group_names)
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.
        files = self.db.objects(protocol=protocol, groups=groups, cls=purposes, **kwargs)
        files = [ReplayPadFile(f) for f in files]
        return files


    #==========================================================================
    def annotations(self, f):
        """
        Return annotations for a given file object ``f``, which is an instance
        of ``ReplayPadFile`` defined in the HLDI of the Replay-Attack DB.
        The ``load()`` method of ``ReplayPadFile`` class (see above)
        returns a video, therefore this method returns bounding-box annotations
        for each video frame. The annotations are returned as dictionary of dictionaries.

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of ``ReplayPadFile`` defined above.

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

            topleft = (frame_annots[2], frame_annots[1])
            bottomright = (frame_annots[2] + frame_annots[4], frame_annots[1] + frame_annots[3])

            annotations[str(fn)] = {'topleft': topleft, 'bottomright': bottomright}

        return annotations

