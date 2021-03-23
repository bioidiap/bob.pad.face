

.. _bob.pad.face.resources:

===========
 Resources
===========

This section contains a listing of all ready-to-use resources you can find in
this package.

---------------------------------


.. _bob.pad.face.resources.databases:

Databases
------------

These configuration files/resources contain parameters of available databases.
The configuration files contain at least the following arguments of the
``bob pad vanilla-pad`` command:

    * ``database``
    * ``protocol``
    * ``groups``


.. _bob.pad.face.resources.databases.replay:

Replay-Attack Database
================================================================================

.. automodule:: bob.pad.face.config.replay_attack
   :members:


.. _bob.pad.face.resources.databases.replay_mobile:

Replay-Mobile Database
================================================================================

.. automodule:: bob.pad.face.config.replay_mobile
   :members:

---------------------------------


.. _bob.pad.face.resources.face_pad:

Available face PAD systems
------------------------------

These configuration files/resources contain parameters of available face PAD
systems/algorithms.
The configuration files contain at least the following arguments of the
``bob pad vanilla-pad`` command:

    * ``pipeline`` containing zero, one, or more Transformers and one Classifier


.. _bob.pad.face.resources.face_pad.lbp_svm_replayattack:

LBP features of facial region + SVM for REPLAY-ATTACK
================================================================================


.. _bob.pad.face.resources.face_pad.qm_svm_replayattack:

Image Quality Measures as features of facial region + SVM for REPLAY-ATTACK
================================================================================




.. _bob.pad.face.resources.face_pad.qm_one_class_gmm:

Image Quality Measures as features of facial region + GMM-based one-class classifier (anomaly detector)
============================================================================================================================


