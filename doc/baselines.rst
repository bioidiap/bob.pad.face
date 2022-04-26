

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


Image Quality Measures as features of facial region + SVM classifier
========================================================================

Detailed description of this PAD pipeline is given at :ref:`bob.pad.face.resources.face_pad.qm_svm_replayattack`.

To run this baseline on the `replayattack`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad replay-attack qm svm-frames \
    --output <PATH_TO_STORE_THE_RESULTS>

.. tip::

    Similarly to the tip above you can run this baseline in parallel with the
    ``--dask-client`` option.

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/qm_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    $ bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/scores-dev.csv  \
    <PATH_TO_STORE_THE_RESULTS>/scores-eval.csv \
    --legends "IQM features of facial region + SVM classifier + REPLAY-ATTACK database" \
    --eval \
    --criterion eer \
    --output <PATH_TO_STORE_THE_RESULTS>/evaluation_report.pdf

The EER/HTER errors for `replayattack`_ database are summarized in the table below:

==============  ================  ==================
..              Development       Evaluation
==============  ================  ==================
APCER (attack)  6.58%             12.96%
APCER_AP        6.58%             12.96%
BPCER           6.58%             0.19%
ACER            6.58%             6.58%
FTA             0.00%             0.00%
FPR             6.58% (395/6000)  12.96% (1037/7999)
FNR             6.58% (79/1200)   0.19% (3/1600)
HTER            6.58%             6.58%
FAR             6.58%             12.96%
FRR             6.58%             0.19%
PRECISION       0.74              0.61
RECALL          0.93              1.00
F1_SCORE        0.83              0.75
AUC             0.98              0.99
AUC-LOG-SCALE   2.91              2.71
==============  ================  ==================


The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_svm_replay_attack.pdf>`

------------



.. _bob.pad.face.baselines.replay_mobile:

Baselines on REPLAY-MOBILE database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the `Replay-Mobile`_ database.
The description of the database-related settings, which are used to run face PAD baselines on the Replay-Mobile is given here :ref:`bob.pad.face.resources.databases.replay_mobile`. To understand the settings in more detail you can check the corresponding configuration file : ``bob/pad/face/config/replay_mobile.py``.



Image Quality Measures as features of facial region + SVM classifier
========================================================================

Detailed description of this PAD pipe-line is given at :ref:`bob.pad.face.resources.face_pad.qm_svm_replayattack`.
Note, that the same PAD pipe-line was used to run experiments on the Replay-Attack database.

To run this baseline on the `Replay-Mobile`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ bob pad vanilla-pad replay-mobile qm svm-frames \
    --output <PATH_TO_STORE_THE_RESULTS>

.. tip::

    Similarly to the tip above you can run this baseline in parallel with the
    ``--dask-client`` option.

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/qm_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    $ bob pad evaluate \
    <PATH_TO_STORE_THE_RESULTS>/scores-dev.csv  \
    <PATH_TO_STORE_THE_RESULTS>/scores-eval.csv \
    --legends "IQM features of facial region + SVM classifier + Replay-Mobile database" \
    --eval \
    --criterion eer \
    --output <PATH_TO_STORE_THE_RESULTS>/evaluation_report.pdf

The EER/HTER errors for the `Replay-Mobile`_ database are summarized in the table below:

===================  ================  =================
..                   Development       Evaluation
===================  ================  =================
APCER (mattescreen)  1.54%             0.16%
APCER (print)        8.16%             9.53%
APCER_AP             8.16%             9.53%
BPCER                4.88%             10.77%
ACER                 6.52%             10.15%
FTA                  0.00%             0.00%
FPR                  4.86% (248/5098)  4.89% (186/3802)
FNR                  4.88% (156/3199)  10.77% (236/2192)
HTER                 4.87%             7.83%
FAR                  4.86%             4.89%
FRR                  4.88%             10.77%
PRECISION            0.92              0.91
RECALL               0.95              0.89
F1_SCORE             0.94              0.90
AUC                  0.99              0.98
AUC-LOG-SCALE        2.54              2.52
===================  ================  =================


The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_svm_replay_mobile.pdf>`

------------


.. include:: links.rst
