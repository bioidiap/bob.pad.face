

.. _bob.pad.face.baselines:


===============================
 Executing Baseline Algorithms
===============================

This section explains how to execute face presentation attack detection (PAD) algorithms implemented
in ``bob.pad.face``.


Running Baseline Experiments
----------------------------

To run the baseline PAD experiments, the ``spoof.py`` script located in ``bin`` directory is used.
To see the description of the script you can type in the console:

.. code-block:: sh

   $ ./bin/verify.py --help

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


Database setups and baselines are encoded using
:ref:`bob.bio.base.configuration-files`, all stored inside the package root, in
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


.. _bob.pad.face.baselines.replay_attack:


------------

Baselines on REPLAY-ATTACK database
--------------------------------------

This section summarizes the results of baseline face PAD experiments on the REPLAY-ATTACK (`replayattack`_) database.


LBP features of facial region + SVM classifier
===================================================

Detailed description of this PAD pipe-line is given at :ref:`bob.pad.face.resources.face_pad.lbp_svm_replayattack`.

To run this baseline on the `replayattack`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ ./bin/spoof.py lbp-svm \
    --database replay --protocol grandtest --groups train dev eval \
    --sub-directory <PATH_TO_STORE_THE_RESULTS>

.. tip::

    If you are in `idiap`_ you can use SGE grid to speed-up the calculations.
    Simply add ``--grid idiap`` argument to the above command. For example:

    .. code-block:: sh

        $ ./bin/spoof.py lbp-svm \
        --database replay --protocol grandtest --groups train dev eval \
        --sub-directory <PATH_TO_STORE_THE_RESULTS> \
        --grid idiap

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/lbp_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    ./bin/evaluate.py \
    --dev-files  <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    --eval-files <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "LBP features of facial region + SVM classifier + REPLAY-ATTACK database" \
    -F 7 \
    --criterion EER \
    --roc <PATH_TO_STORE_THE_RESULTS>/ROC.pdf


The EER/HTER errors for `replayattack`_ database are summarized in the Table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  15.117  |  15.609  |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_lbp_svm_replay_attack.pdf>`


------------

Image Quality Measures as features of facial region + SVM classifier
========================================================================

Detailed description of this PAD pipe-line is given at :ref:`bob.pad.face.resources.face_pad.qm_svm_replayattack`.

To run this baseline on the `replayattack`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ ./bin/spoof.py qm-svm \
    --database replay --protocol grandtest --groups train dev eval \
    --sub-directory <PATH_TO_STORE_THE_RESULTS>

.. tip::

    Similarly to the tip above you can run this baseline in parallel.

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/qm_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    ./bin/evaluate.py \
    --dev-files  <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    --eval-files <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "IQM features of facial region + SVM classifier + REPLAY-ATTACK database" \
    -F 7 \
    --criterion EER \
    --roc <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for `replayattack`_ database are summarized in the Table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  4.321   |  4.570   |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_svm_replay_attack.pdf>`

------------


Frame differences based features (motion analysis) + SVM classifier
========================================================================

Detailed description of this PAD pipe-line is given at :ref:`bob.pad.face.resources.face_pad.frame_diff_svm_replayattack`.

To run this baseline on the `replayattack`_ database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ ./bin/spoof.py frame-diff-svm \
    --database replay --protocol grandtest --groups train dev eval \
    --sub-directory <PATH_TO_STORE_THE_RESULTS>

.. tip::

    Similarly to the tip above you can run this baseline in parallel.

To understand the settings of this baseline PAD experiment you can check the
corresponding configuration file: ``bob/pad/face/config/frame_diff_svm.py``

To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    ./bin/evaluate.py \
    --dev-files  <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-dev  \
    --eval-files <PATH_TO_STORE_THE_RESULTS>/grandtest/scores/scores-eval \
    --legends "10 features for each window in Frame Differences + SVM classifier + REPLAY-ATTACK database" \
    -F 7 \
    --criterion EER \
    --roc <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for `replayattack`_ database are summarized in the Table below:

+-------------------+----------+----------+
|      Protocol     |  EER,\%  |  HTER,\% |
+===================+==========+==========+
|   ``grandtest``   |  11.752  |  13.195  |
+-------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_frame_diff_svm_replay_attack.pdf>`

------------


.. include:: links.rst
