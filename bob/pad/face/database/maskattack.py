#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import bob.io.video
from bob.bio.video import FrameSelector, FrameContainer
from bob.pad.face.database import VideoPadFile  # Used in MsuMfsdPadFile class
from bob.pad.base.database import PadDatabase

class MaskAttackPadFile(VideoPadFile):
    """
    A high level implementation of the File class for the 3DMAD database.
    """

    def __init__(self, f):
        """
        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of the File class defined in the low level db interface
            of the 3DMAD database, in the bob.db.maskattack.models.py file.
        """

        self.f = f
        if f.is_real():
            attack_type = None
        else:
            attack_type = 'mask'

        super(MaskAttackPadFile, self).__init__(
            client_id=f.client_id,
            path=f.path,
            attack_type=attack_type,
            file_id=f.id)

    #==========================================================================
    def load(self, directory=None, extension='.avi', frame_selector=FrameSelector(selection_style='all')):
        """
        Overridden version of the load method defined in the ``VideoPadFile``.

        **Parameters:**

        ``directory`` : :py:class:`str`
            String containing the path to the MSU MFSD database.
            Default: None

        ``extension`` : :py:class:`str`
            Extension of the video files in the MSU MFSD database.
            Note: ``extension`` value is not used in the code of this method.
            Default: None

        ``frame_selector`` : ``FrameSelector``
            The frame selector to use.

        **Returns:**

        ``video_data`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.
        """
        vfilename = self.make_path(directory, extension)
        video = bob.io.video.reader(vfilename)
        video_data_array = video.load()
        
        return frame_selector(video_data_array)


#==============================================================================
class MaskAttackPadDatabase(PadDatabase):
    """
    A high level implementation of the Database class for the 3DMAD database.
    """

    def __init__(
            self,
            protocol=None,  
            original_directory=None,
            original_extension='.avi',
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

        from bob.db.maskattack import Database as LowLevelDatabase

        self.db = LowLevelDatabase()

        # Since the high level API expects different group names than what the low
        # level API offers, you need to convert them when necessary
        self.low_level_group_names = (
            'world', 'dev',
            'test')  # group names in the low-level database interface
        self.high_level_group_names = (
            'train', 'dev',
            'eval')  # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        super(MaskAttackPadDatabase, self).__init__(
            name='maskattack',
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

    #==========================================================================
    def objects(self,
                groups=None,
                protocol=None,
                purposes=None,
                model_ids=None,
                **kwargs):
        """
        This function returns lists of MaskAttackPadFile objects, which fulfill the given restrictions.

        Keyword parameters:

        ``groups`` : :py:class:`str`
            OR a list of strings.
            The groups of which the clients should be returned.
            Usually, groups are one or more elements of ('train', 'dev', 'eval')

        ``protocol`` : :py:class:`str`
            The protocol for which the clients should be retrieved.
            Note: this argument is not used in the code, because ``objects`` method of the
            low-level BD interface of the MSU MFSD doesn't have ``protocol`` argument.

        ``purposes`` : :py:class:`str`
            OR a list of strings.
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.

        ``model_ids``
            This parameter is not supported in PAD databases yet.

        **Returns:**

        ``files`` : [MsuMfsdPadFile]
            A list of MsuMfsdPadFile objects.
        """

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(
            groups, self.low_level_group_names, self.high_level_group_names)
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.

        print("Objects method called with groups = {}, protocol = {}, purposes = {}, model_ids = {}".format(groups, protocol, purposes, model_ids))
        #print("Kwargs -> {}".format(**kwargs))
        #print("Translated groups = {}".frima)
        
        # for training

        # for dev

        # for eval
        lowlevel_purposes = []
        if purposes == 'real':
          lowlevel_purposes = ['trainReal', 'probeReal', 'classifyReal']
        else:
          lowlevel_purposes = ['trainMask', 'probeMask', 'classifyMask']
          
        #if groups == ['world']:
        #    lowlevel_purposes = ['trainMask']
        #  if groups == ['world']:
        #    lowlevel_purposes = ['trainMask']
        #print(lowlevel_purposes)
        files = self.db.objects(sets=groups, purposes=lowlevel_purposes, **kwargs)

        files = [MaskAttackPadFile(f) for f in files]

        return files

    #==========================================================================
    def annotations(self, file):
        """
        Return annotations for a given file object ``f``, which is an instance
        of ``MsuMfsdPadFile`` defined in the HLDI of the MSU MFSD DB.
        The ``load()`` method of ``MsuMfsdPadFile`` class (see above)
        returns a video, therefore this method returns bounding-box annotations
        for each video frame. The annotations are returned as dictionary of dictionaries.

        **Parameters:**

        ``f`` : :py:class:`object`
            An instance of ``MsuMfsdPadFile`` defined above.

        **Returns:**

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box in frame N.
        """
        return None

        #annots = f.f.bbx(
        #    directory=self.original_directory
        #)  # numpy array containing the face bounding box data for each video frame, returned data format described in the f.bbx() method of the low level interface

        #annotations = {}  # dictionary to return

        #for frame_annots in annots:

        #    topleft = (np.int(frame_annots[2]), np.int(frame_annots[1]))
        #    bottomright = (np.int(frame_annots[2] + frame_annots[4]),
        #                   np.int(frame_annots[1] + frame_annots[3]))

        #    annotations[str(np.int(frame_annots[0]))] = {
        #        'topleft': topleft,
        #        'bottomright': bottomright
        #    }

        #return annotations

    #def model_with_ids_protocol(groups=None, protocol=None):
    #  pass
