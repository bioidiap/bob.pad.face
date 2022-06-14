

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
``bob pad run-pipeline`` command:

    * ``database``
    * ``protocol``
    * ``groups``


.. _bob.pad.face.resources.databases.replay_attack:

Replay-Attack Database
================================================================================

.. automodule:: bob.pad.face.config.replay_attack
   :members:


.. _bob.pad.face.resources.databases.replay_mobile:

Replay-Mobile Database
================================================================================

.. automodule:: bob.pad.face.config.replay_mobile
   :members:



.. _bob.pad.face.resources.databases.oulu_npu:

OULU-NPU Database
================================================================================

.. automodule:: bob.pad.face.config.oulu_npu
   :members:



.. _bob.pad.face.resources.databases.swan:

SWAN Database
================================================================================

.. automodule:: bob.pad.face.config.swan
   :members:



.. _bob.pad.face.resources.deep_pix_bis_pad:

Deep Pixel-wise Binary Supervision for Face PAD
================================================================================

.. automodule:: bob.pad.face.config.deep_pix_bis
   :members:



.. _bob.pad.face.resources.face_pad:

Available face PAD systems
------------------------------

These configuration files/resources contain parameters of available face PAD
systems/algorithms.
The configuration files contain at least the following arguments of the
``bob pad run-pipeline`` command:

    * ``pipeline`` containing zero, one, or more Transformers and one Classifier

.. include:: links.rst
