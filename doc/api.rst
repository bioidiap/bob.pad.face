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


Pre-processors
------------------------------

.. automodule:: bob.pad.face.preprocessor


Feature Extractors
------------------------------

.. automodule:: bob.pad.face.extractor


Matching Algorithms
------------------------------

.. automodule:: bob.pad.face.algorithm
