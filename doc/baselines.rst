

.. _bob.pad.face.baselines:


===============================
 Executing Baseline Algorithms
===============================

This section explains how to execute face presentation attack detection (PAD)
algorithms implemented in ``bob.pad.face``.


Running Baseline Experiments
----------------------------

To run the baseline PAD experiments, the ``bob pad run-pipeline`` command is used.
To see the description of the command, you can type in the console:

.. code-block:: sh

   $ bob pad run-pipeline --help

This command is explained in more detail in :ref:`bob.pad.base <bob.pad.base.features>`.

Usually, it is a good idea to have at least verbose level 2 (i.e., calling
``bob pad run-pipeline --verbose --verbose``, or the short version
``bob pad run-pipeline -vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can use an existing or (define a new)
   SGE grid or local host multiprocessing configuration. To run the experiment
   in the Idiap SGE grid, you can simply add the ``--dask-client sge`` command
   line option. To run experiments in parallel on the local machine, add the
   ``--dask-client local-parallel`` option.

   See :any:`this <pipeline_simple_features>` for more
   details on dask configurations.


Database setups and baselines are encoded using
``configuration-files``, all stored inside the package structure, in
the directory ``bob/pad/face/config``. Documentation for each resource
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

   Once the raw data files have been downloaded, unpack the databases carefully
   and take a note of the root directory where they have been unpacked.

   Use the following commands to specify the correct parameters of your dataset
   (see :ref:`bob.pad.face.resources.databases`):

   .. code-block:: sh

      $ bob config set bob.db.replaymobile.directory /path/to/replayattack-database/
      $ bob config set bob.db.replaymobile.extension .mov

   Notice it is rather important to correctly configure the database as
   described above, otherwise ``bob.pad.base`` will not be able to correctly
   load your data.

   Once this step is done, you can proceed with the instructions below.


------------

.. _bob.pad.face.baselines.replay_attack:

Baselines on REPLAY-ATTACK database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the
REPLAY-ATTACK (`replayattack`_) database.
The description of the database-related settings, which are used to run face PAD
baselines on the Replay-Attack is given here
:ref:`bob.pad.face.resources.databases.replay`. To understand the settings in
more detail you can check the corresponding configuration file:
``bob/pad/face/config/replay_attack.py``.




.. _bob.pad.face.baselines.replay_mobile:

Baselines on REPLAY-MOBILE database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the `Replay-Mobile`_ database.
The description of the database-related settings, which are used to run face PAD baselines on the Replay-Mobile is given here :ref:`bob.pad.face.resources.databases.replay_mobile`. To understand the settings in more detail you can check the corresponding configuration file : ``bob/pad/face/config/replay_mobile.py``.



.. include:: links.rst
