

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

This script is explained in more detail in :ref:`bob.pad.base.experiments`.

Usually, it is a good idea to have at least verbose level 2 (i.e., calling
``bob pad --verbose --verbose``, or the short version ``bob pad -vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can define an SGE grid or local host
   (multi-processing) configurations as explained in
   :ref:`running_in_parallel`.

   In short, to run in the Idiap SGE grid, you can simply add the
   ``--dask-client sge`` command line option. To run experiments in parallel on
   the local machine, simply add the ``--dask-client local-parallel`` option.


Database setups and face PAD algorithms are encoded using
``bob.bio.base.configuration-files``, all stored inside the package structure,
in the directory ``bob/pad/face/config``. Documentation for each resource
is available on the section :ref:`bob.pad.face.resources`.

.. warning::

   You **cannot** run experiments just by executing the command line
   instructions described in this guide. You **need first** to procure yourself
   the raw data files that correspond to *each* database used here to correctly
   run experiments with those data. Biometric data is considered private and,
   under EU regulations, cannot be distributed without a consent or license.
   You may consult our :ref:`bob.pad.face.resources.databases` resources
   section for checking currently supported databases and accessing download
   links for the raw data files.

   Once the raw data files have been downloaded, particular attention should be
   given to the directory locations of those. Unpack the databases carefully
   and take note of the root directory where they have been unpacked.

   Then, carefully read the *Databases* section of
   :ref:`bob.pad.base.installation` on how to correctly setup the
   ``~/.bob_bio_databases.txt`` file.

   Use the following command with the appropriate database name (see
   :ref:`bob.pad.face.resources.databases`):

   .. code-block:: sh

      bob config set bob.db.<dbname> /path/to/the/db/folder

   Once this step is done, you can proceed with the execution of the experiment.


.. include:: links.rst


