#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:03:36 2017

High level implementation for the REPLAY-ATTACK database

@author: Olegs Nikisins <olegs.nikisins@idiap.ch>
"""

#==============================================================================

from bob.pad.base.database import PadDatabase

import bob.bio.video # Used in ReplayPadFile class

from bob.pad.base.database import PadFile # Used in ReplayPadFile class

#==============================================================================

class ReplayPadFile(PadFile):
    """
    A high level implementation of the File class for the REPLAY-ATTACK database.
    """

    def __init__(self, f):
        """

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level implementation
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

        super(ReplayPadFile, self).__init__(client_id=f.client, path=f.path,
                                            attack_type=attack_type, file_id=f.id)


    def load(self, directory=None, extension='.mov'):
        """
        Overridden version of the load method defined in the ``bob.db.base.File``.

        **Parameters:**

        ``directory`` : :py:class:`str`
            String containing the path to the Replay database.

        ``extension`` : :py:class:`str`
            Extension of the video files in the Replay database.

        **Returns:**

        ``filtered_image`` : :py:class:`dict`
            A dictionary containing the key-value pairs: "video" key containing the frames data,
            and "bbx" containing the coordinates of the face bounding boxes for each frame.
        """

        path = self.f.make_path(directory=directory, extension=extension) # path to the video file

        frame_selector = bob.bio.video.FrameSelector(selection_style = 'all') # this frame_selector will select all frames from the video file

        video_data = frame_selector(path) # video data

        bbx_data = self.f.bbx(directory=directory) # numpy array containing the face bounding box data for each video frame, returned data format described in the f.bbx() method of the low level interface

        return_dictionary = {}
        return_dictionary["video"] = video_data
        return_dictionary["bbx"] = bbx_data

        return return_dictionary # dictionary containing the face bounding box annotations and video data

#==============================================================================










class ReplayPadDatabase(PadDatabase):

  def __init__(
     self,
     all_files_options={},
     check_original_files_for_existence=False,
     original_directory=None,
     original_extension=None,
     # here I have said grandtest because this is the name of the default
     # protocol for this database
     protocol='grandtest',
     **kwargs):

    from bob.db.replay import Database as LowLevelDatabase
    self.__db = LowLevelDatabase()

    # Since the high level API expects different group names than what the low
    # level API offers, you need to convert them when necessary
    self.low_level_group_names = ('train', 'devel', 'test')
    self.high_level_group_names = ('train', 'dev', 'eval')

    # Always use super to call parent class methods.
    super(ReplayPadDatabase, self).__init__(
       'replay',
       all_files_options,
       check_original_files_for_existence,
       original_directory,
       original_extension,
       protocol,
       **kwargs)

  def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
    # Convert group names to low-level group names here.
    groups = self.convert_names_to_lowlevel(
       groups, self.low_level_group_names, self.high_level_group_names)
    # Since this database was designed for PAD experiments, nothing special
    # needs to be done here.
    files = self.__db.objects(protocol=protocol, groups=groups, cls=purposes, **kwargs)
    files = [ReplayPadFile(f) for f in files]
    return files
