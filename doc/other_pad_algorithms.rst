

.. _bob.pad.face.other_pad_algorithms:


===============================
 Executing Other Algorithms
===============================

This section explains how to execute face presentation attack detection (PAD) algorithms implemented
in ``bob.pad.face``.

.. warning::

   Algorithms introduced in this section might be in the process of publishing.
   Therefore, it is not allowed to publish results introduced in this section
   without the permission of the owner of the package.
   If you are planning to use the results from this section, please contact the
   owner of the package first.
   Please check the ``setup.py`` for contact information.


Running face PAD Experiments
------------------------------

To run the PAD experiments, use the ``bob pad`` command.
To see the description of this command you can type in the console:

.. code-block:: sh

   $ bob pad --help

This script is explained in more detail in :ref:`bob.pad.base`.

Usually, it is a good idea to have at least verbose level 2 (i.e., calling
``bob pad --verbose --verbose``, or the short version ``bob pad -vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can use an existing or (define a new)
   SGE grid or local host multiprocessing configuration. To run the experiment
   in the Idiap SGE grid, you can simply add the ``--dask-client sge`` command
   line option. To run experiments in parallel on the local machine, add the
   ``--dask-client local-parallel`` option.

   See :any:`this <pipeline_simple_features>` for more
   details on dask configurations.


Database setups and face PAD algorithms are encoded using
``bob.bio.base.configuration-files``, all stored inside the package structure,
in the directory ``bob/pad/face/config``. Documentation for each resource
is available on the section :ref:`bob.pad.face.resources`.


.. include:: links.rst
