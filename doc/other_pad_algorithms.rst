

.. _bob.pad.face.other_pad_algorithms:


===============================
 Executing Other Algorithms
===============================

This section explains how to execute face presentation attack detection (PAD) algorithms implemented
in ``bob.pad.face``.

.. warning::

   Algorithms introduced in this section might be in the process of publishing. Therefore, it is not
   allowed to publish results introduced in this section without permission of the owner of the package.
   If you are planning to use the results from this section, please contact the owner of the package first.
   Please check the ``setup.py`` for contact information.


Running face PAD Experiments
------------------------------

To run the PAD experiments, the ``spoof.py`` script located in ``bin`` directory is used.
To see the description of the script you can type in the console:

.. code-block:: sh

   $ spoof.py --help

This script is explained in more detail in :ref:`bob.pad.base.experiments`.

Usually it is a good idea to have at least verbose level 2 (i.e., calling
``spoof.py --verbose --verbose``, or the short version ``spoof.py
-vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can define an SGE grid or local host
   (multi-processing) configurations as explained in
   :ref:`running_in_parallel`.

   In short, to run in the Idiap SGE grid, you can simply add the ``--grid``
   command line option, with grid configuration parameters. To run experiments in parallel on
   the local machine, simply add a ``--parallel <N>`` option, where ``<N>``
   specifies the number of parallel jobs you want to execute.


Database setups and face PAD algorithms are encoded using
``bob.bio.base.configuration-files``, all stored inside the package root, in
the directory ``bob/pad/face/config``. Documentation for each resource
is available on the section :ref:`bob.pad.face.resources`.

.. warning::

   You **cannot** run experiments just by executing the command line
   instructions described in this guide. You **need first** to procure yourself
   the raw data files that correspond to *each* database used here in order to
   correctly run experiments with those data. Biometric data is considered
   private date and, under EU regulations, cannot be distributed without a
   consent or license. You may consult our
   :ref:`bob.pad.face.resources.databases` resources section for checking
   currently supported databases and accessing download links for the raw data
   files.

   Once the raw data files have been downloaded, particular attention should be
   given to the directory locations of those. Unpack the databases carefully
   and annotate the root directory where they have been unpacked.

   Then, carefully read the *Databases* section of
   :ref:`bob.pad.base.installation` on how to correctly setup the
   ``~/.bob_bio_databases.txt`` file.

   Use the following keywords on the left side of the assignment (see
   :ref:`bob.pad.face.resources.databases`):

   .. code-block:: text

      [YOUR_REPLAY_ATTACK_DIRECTORY] = /complete/path/to/replayattack-database/

   Notice it is rather important to use the strings as described above,
   otherwise ``bob.pad.base`` will not be able to correctly load your images.

   Once this step is done, you can proceed with the instructions below.


.. include:: links.rst


