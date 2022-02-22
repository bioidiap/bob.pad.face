.. vim: set fileencoding=utf-8 :

.. _bob.pad.face.api:

============
 Python API
============

This section lists all the functionality available in this library allowing to run face PAD experiments.


Database Interfaces
------------------------------

REPLAY-ATTACK Database
========================

.. autofunction:: bob.pad.face.database.replay_attack.ReplayAttackPadDatabase

REPLAY-MOBILE Database
========================

.. autofunction:: bob.pad.face.database.replay_mobile.ReplayMobilePadDatabase



Transformers
------------

Pre-processors
==============

.. automodule:: bob.pad.face.preprocessor


Feature Extractors
==================

.. automodule:: bob.pad.face.extractor



Utilities
---------

.. autosummary::
   bob.pad.face.utils.bbx_cropper
   bob.pad.face.utils.blocks
   bob.pad.face.utils.blocks_generator
   bob.pad.face.utils.color_augmentation
   bob.pad.face.utils.frames
   bob.pad.face.utils.min_face_size_normalizer
   bob.pad.face.utils.number_of_frames
   bob.pad.face.utils.scale_face
   bob.pad.face.utils.the_giant_video_loader
   bob.pad.face.utils.yield_faces


.. automodule:: bob.pad.face.utils
