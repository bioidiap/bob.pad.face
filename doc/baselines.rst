.. _bob.pad.face.baselines:


===============================
 Executing Baseline Algorithms
===============================

This section explains how to execute face presentation attack detection (PAD).

Running Baseline Experiments
----------------------------

To run the baseline PAD experiments, the ``bob pad run-pipeline`` command is used.
To see the description of the command, you can type in the console:

.. code-block:: sh

   $ bob pad run-pipeline --help

This command is explained in more detail in :ref:`bob.pad.base <bob.pad.base.pipeline_intro>`.

Usually, it is a good idea to have at least verbose level 2 (i.e., calling
``bob pad run-pipeline --verbose --verbose``, or the short version
``bob pad run-pipeline -vv``).

.. note:: **Running in Parallel**

   To run the experiments in parallel, you can use an existing or (define a new)
   SGE grid or local host multiprocessing configuration. To run the experiment
   in the Idiap SGE grid, you can simply add the ``--dask-client sge`` command
   line option. To run experiments in parallel on the local machine, add the
   ``--dask-client local-parallel`` option.

.. note::

   If you run out of memory, you can try to reduce the dask partition size
   by setting the ``--dask-partition-size`` option.

Database setups and baselines are encoded using ``configuration-files``, all
stored inside the package structure, in the directory ``bob/pad/face/config``.
Documentation for each resource is available on the section
:ref:`bob.pad.face.resources`.

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

      $ bob config set bob.db.replay_mobile.directory /path/to/replaymobile-database/

   Notice it is rather important to correctly configure the database as
   described above, otherwise ``bob.pad.base`` will not be able to correctly
   load your data.

   Once this step is done, you can proceed with the instructions below.


------------

.. _bob.pad.face.baselines.replay_attack:

Baselines on REPLAY-ATTACK database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the
REPLAY-ATTACK (`replay-attack`_) database. The description of the
database-related settings, which are used to run face PAD baselines on the
Replay-Attack is given here :ref:`bob.pad.face.resources.databases.replay_attack`. To
understand the settings in more detail you can check the corresponding
configuration file: ``bob/pad/face/config/replay_attack.py``.

Running Baseline
~~~~~~~~~~~~~~~~~~~~~

The baseline is run by invoking the command below

.. code-block:: sh

   $ bob pad run-pipeline -vvv replay-attack <BASELINE> --output <OUTPUT> --dask-client <CLIENT>

where ``<BASELINE>`` can be any of the following: ``lbp-svm``, ``deep-pix-bis``.
This baseline reports scores per frame. To obtain scores per video, you can run::

   $ bob pad finalize-scores -vvv <OUTPUT>/scores-{dev,eval}.csv

Finally, you can evaluate this baseline using::

   $ bob pad metrics -vvv --eval <OUTPUT>/scores-{dev,eval}.csv

LBP-SVM Baseline
~~~~~~~~~~~~~~~~~~~~~

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set ``<OUTPUT>/scores-dev.csv`: -1.042440e+00
   ==============  ==============  ==============
   ..              Development     Evaluation
   ==============  ==============  ==============
   APCER (attack)  20.7%           19.5%
   APCER_AP        20.7%           19.5%
   BPCER           20.0%           13.8%
   ACER            20.3%           16.6%
   FTA             0.0%            0.0%
   FPR             20.7% (62/300)  19.5% (78/400)
   FNR             20.0% (12/60)   13.8% (11/80)
   HTER            20.3%           16.6%
   FAR             20.7%           19.5%
   FRR             20.0%           13.8%
   PRECISION       0.4             0.5
   RECALL          0.8             0.9
   F1_SCORE        0.6             0.6
   AUC             0.9             0.9
   AUC-LOG-SCALE   1.4             1.5
   ==============  ==============  ==============


Deep-Pix-BiS Baseline
~~~~~~~~~~~~~~~~~~~~~
(see :ref:`bob.pad.face.resources.deep_pix_bis_pad`)

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set `<OUTPUT>/scores-dev.csv`: 1.919391e-01
   ==============  ==============  ===============
   ..              Development     Evaluation
   ==============  ==============  ===============
   APCER (attack)  32.3%           34.0%
   APCER_AP        32.3%           34.0%
   BPCER           31.7%           27.5%
   ACER            32.0%           30.8%
   FTA             0.0%            0.0%
   FPR             32.3% (97/300)  34.0% (136/400)
   FNR             31.7% (19/60)   27.5% (22/80)
   HTER            32.0%           30.8%
   FAR             32.3%           34.0%
   FRR             31.7%           27.5%
   PRECISION       0.3             0.3
   RECALL          0.7             0.7
   F1_SCORE        0.4             0.4
   AUC             0.7             0.7
   AUC-LOG-SCALE   1.5             1.4
   ==============  ==============  ===============


.. _bob.pad.face.baselines.replay_mobile:

Baselines on REPLAY-MOBILE database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the
`Replay-Mobile`_ database. The description of the database-related settings,
which are used to run face PAD baselines on the Replay-Mobile is given here
:ref:`bob.pad.face.resources.databases.replay_mobile`. To understand the
settings in more detail you can check the corresponding configuration file :
``bob/pad/face/config/replay_mobile.py``.


Running Baseline
~~~~~~~~~~~~~~~~~~~~~

The baseline is run by invoking the command below

.. code-block:: sh

   $ bob pad run-pipeline -vvv replay-mobile <BASELINE> --output <OUTPUT> --dask-client <CLIENT>

where ``<BASELINE>`` can be any of the following: ``lbp-svm``, ``deep-pix-bis``.
This baseline reports scores per frame. To obtain scores per video, you can run::

   $ bob pad finalize-scores -vvv <OUTPUT>/scores-{dev,eval}.csv

Finally, you can evaluate this baseline using::

   $ bob pad metrics -vvv --eval <OUTPUT>/scores-{dev,eval}.csv


LBP-SVM Baseline
~~~~~~~~~~~~~~~~~~~~~

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set `<OUTPUT>/scores-dev.csv`: -5.045229e-01
   ===================  ==============  ==============
   ..                   Development     Evaluation
   ===================  ==============  ==============
   APCER (mattescreen)  7.0%            7.3%
   APCER (print)        9.4%            3.1%
   APCER_AP             9.4%            7.3%
   BPCER                8.1%            0.9%
   ACER                 8.8%            4.1%
   FTA                  0.0%            0.0%
   FPR                  8.2% (21/256)   5.2% (10/192)
   FNR                  8.1% (13/160)   0.9% (1/110)
   HTER                 8.2%            3.1%
   FAR                  8.2%            5.2%
   FRR                  8.1%            0.9%
   PRECISION            0.9             0.9
   RECALL               0.9             1.0
   F1_SCORE             0.9             1.0
   AUC                  1.0             1.0
   AUC-LOG-SCALE        2.2             2.2
   ===================  ==============  ==============


Deep-Pix-BiS Baseline
~~~~~~~~~~~~~~~~~~~~~

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set `<OUTPUT>/scores-dev.csv`: 4.051177e-01
   ===================  ==============  ==============
   ..                   Development     Evaluation
   ===================  ==============  ==============
   APCER (mattescreen)  4.7%            8.3%
   APCER (print)        15.6%           18.8%
   APCER_AP             15.6%           18.8%
   BPCER                10.0%           10.9%
   ACER                 12.8%           14.8%
   FTA                  0.0%            0.0%
   FPR                  10.2% (26/256)  13.5% (26/192)
   FNR                  10.0% (16/160)  10.9% (12/110)
   HTER                 10.1%           12.2%
   FAR                  10.2%           13.5%
   FRR                  10.0%           10.9%
   PRECISION            0.8             0.8
   RECALL               0.9             0.9
   F1_SCORE             0.9             0.8
   AUC                  1.0             1.0
   AUC-LOG-SCALE        2.0             1.8
   ===================  ==============  ==============


.. _bob.pad.face.baselines.oulu_npu:

Baselines on OULU-NPU database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the
`OULU-NPU`_ database. The description of the database-related settings,
which are used to run face PAD baselines on the OULU-NPU is given here
:ref:`bob.pad.face.resources.databases.oulu_npu`. To understand the
settings in more detail you can check the corresponding configuration file :
``bob/pad/face/config/oulu_npu.py``.


Running Baseline
~~~~~~~~~~~~~~~~~~~~~

The baseline is run by invoking the command below

.. code-block:: sh

   $ bob pad run-pipeline -vvv oulu-npu <BASELINE> --output <OUTPUT> --dask-client <CLIENT>

where ``<BASELINE>`` can be any of the following: ``lbp-svm``, ``deep-pix-bis``.
This baseline reports scores per frame. To obtain scores per video, you can run::

   $ bob pad finalize-scores -vvv <OUTPUT>/scores-{dev,eval}.csv

Finally, you can evaluate this baseline using::

   $ bob pad metrics -vvv --eval <OUTPUT>/scores-{dev,eval}.csv


LBP-SVM Baseline
~~~~~~~~~~~~~~~~~~~~~

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set `<OUTPUT>/scores-dev.csv`: 6.161214e-02
   ======================  =============  ============
   ..                      Development    Evaluation
   ======================  =============  ============
   APCER (print/1)         13.3%          10.0%
   APCER (print/2)         11.1%          6.7%
   APCER (video_replay/1)  4.4%           5.8%
   APCER (video_replay/2)  8.3%           11.7%
   APCER_AP                13.3%          11.7%
   BPCER                   9.4%           57.5%
   ACER                    11.4%          34.6%
   FTA                     0.0%           0.0%
   FPR                     9.3% (67/720)  8.5% (41/480)
   FNR                     9.4% (17/180)  57.5% (69/120)
   HTER                    9.4%           33.0%
   FAR                     9.3%           8.5%
   FRR                     9.4%           57.5%
   PRECISION               0.7            0.6
   RECALL                  0.9            0.4
   F1_SCORE                0.8            0.5
   AUC                     1.0            0.8
   AUC-LOG-SCALE           2.1            1.1
   ======================  =============  ============


Deep-Pix-BiS Baseline
~~~~~~~~~~~~~~~~~~~~~

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set `<OUTPUT>/scores-dev.csv`: 4.326179e-01
   ======================  =============  ============
   ..                      Development    Evaluation
   ======================  =============  ============
   APCER (print/1)         0.6%           1.7%
   APCER (print/2)         0.0%           4.2%
   APCER (video_replay/1)  1.7%           0.0%
   APCER (video_replay/2)  0.0%           0.8%
   APCER_AP                1.7%           4.2%
   BPCER                   0.6%           0.0%
   ACER                    1.1%           2.1%
   FTA                     0.0%           0.0%
   FPR                     0.6% (4/720)   1.7% (8/480)
   FNR                     0.6% (1/180)   0.0% (0/120)
   HTER                    0.6%           0.8%
   FAR                     0.6%           1.7%
   FRR                     0.6%           0.0%
   PRECISION               1.0            0.9
   RECALL                  1.0            1.0
   F1_SCORE                1.0            1.0
   AUC                     1.0            1.0
   AUC-LOG-SCALE           2.9            2.7
   ======================  =============  ============


.. _bob.pad.face.baselines.swan:

Baselines on SWAN database
--------------------------

This section summarizes the results of baseline face PAD experiments on the
`SWAN`_ database. The description of the database-related settings,
which are used to run face PAD baselines on the SWAN is given here
:ref:`bob.pad.face.resources.databases.swan`. To understand the
settings in more detail you can check the corresponding configuration file :
``bob/pad/face/config/swan.py``.


Running Baseline
~~~~~~~~~~~~~~~~~~~~~

The baseline is run by invoking the command below

.. code-block:: sh

   $ bob pad run-pipeline -vvv swan <BASELINE> --output <OUTPUT> --dask-client <CLIENT>

where ``<BASELINE>`` can be any of the following: ``lbp-svm``, ``deep-pix-bis``.
This baseline reports scores per frame. To obtain scores per video, you can run::

   $ bob pad finalize-scores -vvv <OUTPUT>/scores-{dev,eval}.csv

Finally, you can evaluate this baseline using::

   $ bob pad metrics -vvv --eval <OUTPUT>/scores-{dev,eval}.csv


LBP-SVM Baseline
~~~~~~~~~~~~~~~~~~~~~

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set `<OUTPUT>/scores-dev.csv`: 9.408851e-02
   ======================  =============  ============
   ..                      Development    Evaluation
   ======================  =============  ============
   APCER (PA.F.1)          6.7%           11.1%
   APCER (PA.F.5)          0.8%           0.8%
   APCER (PA.F.6)          11.2%          10.8%
   APCER_AP                11.2%          11.1%
   BPCER                   6.0%           25.2%
   ACER                    8.6%           18.2%
   FTA                     0.0%           0.0%
   FPR                     6.0% (30/502)  6.0% (45/749)
   FNR                     6.0% (18/300)  25.2% (568/2250)
   HTER                    6.0%           15.6%
   FAR                     6.0%           6.0%
   FRR                     6.0%           25.2%
   PRECISION               0.9            1.0
   RECALL                  0.9            0.7
   F1_SCORE                0.9            0.8
   AUC                     1.0            1.0
   AUC-LOG-SCALE           2.4            2.0
   ======================  =============  ============


Deep-Pix-BiS Baseline
~~~~~~~~~~~~~~~~~~~~~

The metrics for this baseline should give you::

   [Min. criterion: EER ] Threshold on Development set `<OUTPUT>/scores-dev.csv`: 4.867174e-01
   ==============  ==============  ================
   ..              Development     Evaluation
   ==============  ==============  ================
   APCER (PA.F.1)  60.0%           51.1%
   APCER (PA.F.5)  0.8%            2.8%
   APCER (PA.F.6)  16.8%           16.3%
   APCER_AP        60.0%           51.1%
   BPCER           11.7%           21.8%
   ACER            35.8%           36.5%
   FTA             0.0%            0.0%
   FPR             11.8% (59/502)  11.9% (89/749)
   FNR             11.7% (35/300)  21.8% (491/2250)
   HTER            11.7%           16.9%
   FAR             11.8%           11.9%
   FRR             11.7%           21.8%
   PRECISION       0.8             1.0
   RECALL          0.9             0.8
   F1_SCORE        0.8             0.9
   AUC             1.0             0.9
   AUC-LOG-SCALE   2.0             1.6
   ==============  ==============  ================

.. include:: links.rst
