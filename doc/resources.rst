

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
The configuration files contain at least the following arguments of the ``spoof.py`` script:

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


.. _bob.pad.face.resources.databases.msu_mfsd:

MSU MFSD Database
================================================================================

.. automodule:: bob.pad.face.config.msu_mfsd
   :members:


.. _bob.pad.face.resources.databases.aggregated_db:

Aggregated Database
================================================================================

.. automodule:: bob.pad.face.config.aggregated_db
   :members:

MIFS Database
================================================================================

.. automodule:: bob.pad.face.config.mifs
  :members:

---------------------------------


.. _bob.pad.face.resources.face_pad:

Available face PAD systems
------------------------------

These configuration files/resources contain parameters of available face PAD systems/algorithms.
The configuration files contain at least the following arguments of the ``spoof.py`` script:

    * ``sub_directory``
    * ``preprocessor``
    * ``extractor``
    * ``algorithm``


.. _bob.pad.face.resources.face_pad.lbp_svm_replayattack:

LBP features of facial region + SVM for REPLAY-ATTACK
================================================================================

.. automodule:: bob.pad.face.config.lbp_svm
   :members:


.. _bob.pad.face.resources.face_pad.qm_svm_replayattack:

Image Quality Measures as features of facial region + SVM for REPLAY-ATTACK
================================================================================

.. automodule:: bob.pad.face.config.qm_svm
   :members:


.. _bob.pad.face.resources.face_pad.frame_diff_svm_replayattack:

Frame differences based features (motion analysis) + SVM for REPLAY-ATTACK
================================================================================

.. automodule:: bob.pad.face.config.frame_diff_svm
   :members:


.. _bob.pad.face.resources.face_pad.lbp_svm_aggregated_db:

LBP features of facial region + SVM for Aggregated Database
===================================================================================

.. automodule:: bob.pad.face.config.lbp_svm_aggregated_db
   :members:


.. _bob.pad.face.resources.face_pad.qm_svm_aggregated_db:

Image Quality Measures as features of facial region + SVM for Aggregated Database
===================================================================================

.. automodule:: bob.pad.face.config.qm_svm_aggregated_db
   :members:


.. _bob.pad.face.resources.face_pad.frame_diff_svm_aggregated_db:

Frame differences based features (motion analysis) + SVM for Aggregated Database
===================================================================================

.. automodule:: bob.pad.face.config.frame_diff_svm_aggregated_db
   :members:


.. _bob.pad.face.resources.face_pad.qm_lr:

Image Quality Measures as features of facial region + Logistic Regression
============================================================================================================================

.. automodule:: bob.pad.face.config.qm_lr
   :members:


.. _bob.pad.face.resources.face_pad.qm_one_class_gmm:

Image Quality Measures as features of facial region + GMM-based one-class classifier (anomaly detector)
============================================================================================================================

.. automodule:: bob.pad.face.config.qm_one_class_gmm
   :members:


.. _bob.pad.face.resources.face_pad.qm_one_class_svm_aggregated_db:

Image Quality Measures as features of facial region + one-class SVM classifier (anomaly detector) for Aggregated Database
============================================================================================================================

.. automodule:: bob.pad.face.config.qm_one_class_svm_aggregated_db
   :members:


