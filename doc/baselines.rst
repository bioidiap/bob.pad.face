

.. _bob.pad.face.baselines:


===============================
 Executing Baseline Algorithms
===============================

This section explains how to execute face presentation attack detection (PAD)
algorithms implemented in ``bob.pad.face``.


Running Baseline Experiments
----------------------------

To run the baseline PAD experiments, the ``bob pad vanilla-pad`` command is used.
To see the description of the command, you can type in the console:

.. code-block:: sh

   $ bob pad vanilla-pad --help

This command is explained in more detail in :ref:`bob.pad.base <bob.pad.base.vanilla_pad_features>`.

Usually, it is a good idea to have at least verbose level 2 (i.e., calling
``bob pad vanilla-pad --verbose --verbose``, or the short version
``bob pad vanilla-pad -vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can use an existing or (define a new)
   SGE grid or local host multiprocessing configuration. To run the experiment
   in the Idiap SGE grid, you can simply add the ``--dask-client sge`` command
   line option. To run experiments in parallel on the local machine, add the 
   ``--dask-client local-parallel`` option.

   See :ref:`this <bob.bio.base.vanilla_biometrics_advanced_features>` for more
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


LBP features of facial region + SVM classifier
===================================================

Detailed description of this PAD pipe-line is given at
:ref:`bob.pad.face.resources.face_pad.lbp_svm_replayattack`.

To run this baseline on the `replayattack`_ database, using the ``grandtest``
protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad replay-attack lbp svm-frames -o <PATH_TO_STORE_THE_RESULTS>

.. tip::

    If you are at `idiap`_ you can use the SGE grid to speed-up the calculations.
    Simply add the ``--dask-client sge`` (or ``-l sge``) argument to the above
    command. For example:

    .. code-block:: sh

        $ bob pad vanilla-pad replay-attack lbp svm-frames \
        --output <PATH_TO_STORE_THE_RESULTS> \
        --dask-client idiap

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/lbp_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS>/scores-eval \
    --legends "LBP features of facial region + SVM classifier + REPLAY-ATTACK database" \
    -e \
    --criterion eer \
    -o <PATH_TO_STORE_THE_RESULTS>/ROC.pdf


The error rates for `replayattack`_ database are summarized in the table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  15.117  |  15.609  |
+-------------------+----------+----------+

The ROC curves for this particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_lbp_svm_replay_attack.pdf>`

------------


Image Quality Measures as features of facial region + SVM classifier
========================================================================

Detailed description of this PAD pipeline is given at :ref:`bob.pad.face.resources.face_pad.qm_svm_replayattack`.

To run this baseline on the `replayattack`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad replay-attack qm svm-frames \
    --sub-directory <PATH_TO_STORE_THE_RESULTS>

.. tip::

    Similarly to the tip above you can run this baseline in parallel.

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/qm_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "IQM features of facial region + SVM classifier + REPLAY-ATTACK database" \
    -e \
    --criterion eer \
    -o <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for `replayattack`_ database are summarized in the table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  3.987   |  4.571   |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_svm_replay_attack.pdf>`

------------



.. _bob.pad.face.baselines.replay_mobile:

Baselines on REPLAY-MOBILE database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the `Replay-Mobile`_ database.
The description of the database-related settings, which are used to run face PAD baselines on the Replay-Mobile is given here :ref:`bob.pad.face.resources.databases.replay_mobile`. To understand the settings in more detail you can check the corresponding configuration file : ``bob/pad/face/config/replay_mobile.py``.


LBP features of facial region + SVM classifier
========================================================================

Detailed description of this PAD pipe-line is given at :ref:`bob.pad.face.resources.face_pad.lbp_svm_replayattack`.
Note, that the same PAD pipe-line was used to run experiments on the Replay-Attack database.

To run this baseline on the `Replay-Mobile`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad replay-mobile lbp svm_frame \
    --output <PATH_TO_STORE_THE_RESULTS>

.. tip::

    Similarly to the tip above you can run this baseline in parallel.

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/lbp_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "LBP features of facial region + SVM classifier + Replay-Mobile database" \
    -e \
    --criterion eer \
    -o <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for the `Replay-Mobile`_ database are summarized in the table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  13.814  |  17.174  |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_lbp_svm_replay_mobile.pdf>`

------------


Image Quality Measures as features of facial region + SVM classifier
========================================================================

Detailed description of this PAD pipe-line is given at :ref:`bob.pad.face.resources.face_pad.qm_svm_replayattack`.
Note, that the same PAD pipe-line was used to run experiments on the Replay-Attack database.

To run this baseline on the `Replay-Mobile`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad replay-mobile qm svm-frames \
    --output <PATH_TO_STORE_THE_RESULTS>

.. tip::

    Similarly to the tip above you can run this baseline in parallel.

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/qm_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "IQM features of facial region + SVM classifier + Replay-Mobile database" \
    -e \
    --criterion eer \
    -o <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for the `Replay-Mobile`_ database are summarized in the table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  1.747   |  4.074   |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_svm_replay_mobile.pdf>`

------------




.. _bob.pad.face.baselines.other_db:

Baselines on other databases
--------------------------------------

This section summarizes the results of baseline face PAD experiments on other databases.

------------


MIFS database + LBP features of facial region + SVM classifier
========================================================================

To run this baseline on the MIFS database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad mifs lbp svm-frames \
    --output <PATH_TO_STORE_THE_RESULTS>

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "LBP features of facial region + SVM classifier + MIFS database" \
    -e \
    --criterion eer \
    -o <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for the MIFS database are summarized in the table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  27.143  |  32.353  |
+-------------------+----------+----------+

------------


MIFS database + Image Quality Measures as features of facial region + SVM classifier
========================================================================================

To run this baseline on the MIFS database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad mifs qm svm-frames \
    --output <PATH_TO_STORE_THE_RESULTS>

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "IQM features of facial region + SVM classifier + MIFS database" \
    -e \
    --criterion eer \
    -o <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for the MIFS database are summarized in the table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  34.286  |  28.676  |
+-------------------+----------+----------+

------------


.. include:: links.rst
