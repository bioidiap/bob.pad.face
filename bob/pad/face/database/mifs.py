#!/usr/bin/env python2
# -*- coding: utf-8 -*-


#==============================================================================

import bob.bio.video # Used in MIFSPadFile class
import bob.io.base
import numpy as np

from bob.pad.base.database import PadFile # Used in ReplayPadFile class

from bob.pad.base.database import FileListPadDatabase


#==============================================================================

class MIFSPadFile(PadFile):
    """
    A high level implementation of the File class for the MIFS database.
    """

    def __init__(self, client_id, path, attack_type=None, file_id=None):
        super(MIFSPadFile, self).__init__(client_id, path, attack_type, file_id)

    #==========================================================================
    def load(self, directory=None, extension=None):
        #path = self.make_path(directory, extension)
        path = self.make_path(directory=directory, extension=extension) # path to the file
        frame_selector = bob.bio.video.FrameSelector(selection_style = 'all') # this frame_selector will select all frames from the video file

        data = bob.io.base.load(path)
        data = np.expand_dims(data, axis=0) # upgade to 4D (video)
        video_data = frame_selector(data) # video data

        return video_data # video data


#==============================================================================
class MIFSPadDatabase(FileListPadDatabase):
    """
    A high level implementation of the Database class for the MIFS database.
    """

    def __init__(
        self,
        protocol='grandtest', # grandtest is the default protocol for this database
        original_directory='[MIFS_DATABASE_DIRECTORY]',
        original_extension='.jpg',
        **kwargs):

        #from bob.db.replay import Database as LowLevelDatabase

        #self.db = LowLevelDatabase()

        # Since the high level API expects different group names than what the low
        # level API offers, you need to convert them when necessary
        #self.low_level_group_names = ('train', 'devel', 'test') # group names in the low-level database interface
        #self.high_level_group_names = ('train', 'dev', 'eval') # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        #super(ReplayPadDatabase, self).__init__(
        #    name = 'replay',
        #    protocol = protocol,
        #    original_directory = original_directory,
        #    original_extension = original_extension,
        #    **kwargs)

        from pkg_resources import resource_filename
        folder = resource_filename(__name__, '../lists/mifs')
        super(MIFSPadDatabase, self).__init__(folder, 'mifs',
                                            pad_file_class=MIFSPadFile,
                                            protocol = protocol,
                                            original_directory=original_directory,
                                            original_extension=original_extension)

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

        path_to_file    = self.original_directory + f.path
        face_path       = path_to_file[:-4] + '.face'
        file_handle     = open(face_path, 'r')
        line            = file_handle.readline()
        bbox            = [int(x) for x in line.split()]

        annotations = {} # dictionary to return

        topleft = (bbox[1], bbox[0])
        bottomright = (bbox[1] + bbox[3], bbox[0] + bbox[2])

        annotations['0'] = {'topleft': topleft, 'bottomright': bottomright}

        return annotations
