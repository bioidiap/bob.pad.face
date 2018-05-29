.. vim: set fileencoding=utf-8 :

.. _bob.pad.face.api:

============
 Python API
============

This section lists all the functionality available in this library allowing to run face PAD experiments.


Database Interfaces
------------------------------

Base classes
============

.. autoclass:: bob.pad.face.database.VideoPadFile


REPLAY-ATTACK Database
========================

.. autoclass:: bob.pad.face.database.replay.ReplayPadFile
.. autoclass:: bob.pad.face.database.replay.ReplayPadDatabase

REPLAY-MOBILE Database
========================

.. autoclass:: bob.pad.face.database.replay_mobile.ReplayMobilePadFile
.. autoclass:: bob.pad.face.database.replay_mobile.ReplayMobilePadDatabase

MSU MFSD Database
========================

.. autoclass:: bob.pad.face.database.msu_mfsd.MsuMfsdPadFile
.. autoclass:: bob.pad.face.database.msu_mfsd.MsuMfsdPadDatabase

Aggregated Database
========================

.. autoclass:: bob.pad.face.database.aggregated_db.AggregatedDbPadFile
.. autoclass:: bob.pad.face.database.aggregated_db.AggregatedDbPadDatabase

MIFS Database
========================

.. autoclass:: bob.pad.face.database.mifs.MIFSPadFile
.. autoclass:: bob.pad.face.database.mifs.MIFSPadDatabase


Pre-processors
------------------------------

.. automodule:: bob.pad.face.preprocessor


Feature Extractors
------------------------------

.. automodule:: bob.pad.face.extractor


Matching Algorithms
------------------------------

.. automodule:: bob.pad.base.algorithm


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
