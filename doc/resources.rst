

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

These configuration files/resources contain entry points for the ``--database`` command line argument of the
``spoof.py`` script.


.. _bob.pad.face.resources.databases.replay:

Replay-Attack Database
================================================================================

.. automodule:: bob.pad.face.config.database.replay
   :members:


.. _bob.pad.face.resources.databases.replay_mobile:

Replay-Mobile Database
================================================================================

.. automodule:: bob.pad.face.config.database.replay_mobile
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
