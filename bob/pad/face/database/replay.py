#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Amir Mohammadi <amir.mohammadi@idiap.ch>
# Fri 10 Jun 2016 16:48:44 CEST

"""Replay attack database implementation as bob.bio.db.Database"""

from bob.pad.voice.database import PadVoiceFile
from bob.pad.base.database import PadDatabase


class ReplayPadFile(PadVoiceFile):

  def __init__(self, f):
    """
    Initializes this File object with our own File equivalent
    """

    self.__f = f
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
