

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


------------


.. _bob.pad.face.other_pad_algorithms.aggregated_db:

Anomaly detection based PAD on Aggregated Database
--------------------------------------------------------

This section summarizes the results of *anomaly detection* based face PAD experiments on the Aggregated Database.
The description of the database-related settings, which are used to run face PAD algorithms on the Aggregated Db is given here :ref:`bob.pad.face.resources.databases.aggregated_db`. To understand the settings in more details you can check the corresponding configuration file : ``bob/pad/face/config/aggregated_db.py``.

------------


Results for *grandtest* protocol
========================================================================

This section summarizes the evaluation results on the **grandtest** protocol of the Aggregated database for the following face PAD algorithms (for more details click on the corresponding algorithm):

- :ref:`bob.pad.face.resources.face_pad.qm_one_class_gmm`,
- :ref:`bob.pad.face.resources.face_pad.qm_one_class_svm_aggregated_db`,
- :ref:`bob.pad.face.resources.face_pad.qm_lr`,
- :ref:`bob.pad.face.resources.face_pad.qm_svm_aggregated_db`.

For a more detailed understanding of above pipe-lines you can also check corresponding configuration files:

- ``bob/pad/face/config/qm_one_class_gmm.py``,
- ``bob/pad/face/config/qm_one_class_svm_aggregated_db.py``,
- ``bob/pad/face/config/qm_lr.py``,
- ``bob/pad/face/config/qm_svm_aggregated_db.py``.

To run above algorithms on the :ref:`bob.pad.face.resources.databases.aggregated_db` database, using the ``grandtest`` protocol, execute the following:

.. code-block:: sh

    $ spoof.py aggregated-db qm-one-class-gmm \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_1>

    $ spoof.py aggregated-db qm-one-class-svm-aggregated-db \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_2>

    $ spoof.py aggregated-db qm-lr \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_3>

    $ spoof.py aggregated-db qm-svm-aggregated-db \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_4>

.. tip::

    If you are in `idiap`_ you can use SGE grid to speed-up the calculations.
    Simply add ``--grid idiap`` argument to the above command. For example:


To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    ./bin/evaluate.py \
    --dev-files \
    <PATH_TO_STORE_THE_RESULTS_1>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_2>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_3>/grandtest/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_4>/grandtest/scores/scores-dev  \
    --eval-files \
    <PATH_TO_STORE_THE_RESULTS_1>/grandtest/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_2>/grandtest/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_3>/grandtest/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_4>/grandtest/scores/scores-eval \
    --legends \
    "IQM + one-class GMM + Aggregated Db" \
    "IQM + one-class SVM + Aggregated Db" \
    "IQM + two-class LR  + Aggregated Db" \
    "IQM + two-class SVM + Aggregated Db" \
    -F 7 \
    --criterion EER \
    --roc <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for the :ref:`bob.pad.face.resources.databases.aggregated_db` database are summarized in the Table below:

+------------------------+----------+----------+
|      Algorithm         |  EER,\%  |  HTER,\% |
+========================+==========+==========+
|   IQM + one-class GMM  |  19.336  |  20.769  |
+------------------------+----------+----------+
|   IQM + one-class SVM  |  28.137  |  34.776  |
+------------------------+----------+----------+
|   IQM + two-class LR   |  10.354  |  11.856  |
+------------------------+----------+----------+
|   IQM + two-class SVM  |  12.710  |  15.253  |
+------------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_anomaly_detection_aggr_db_grandtest.pdf>`

------------


Results for *photo-photo-video* protocol
========================================================================

This section summarizes the evaluation results on the **photo-photo-video** protocol of the Aggregated database for the following face PAD algorithms (for more details click on the corresponding algorithm):

- :ref:`bob.pad.face.resources.face_pad.qm_one_class_gmm`,
- :ref:`bob.pad.face.resources.face_pad.qm_one_class_svm_aggregated_db`,
- :ref:`bob.pad.face.resources.face_pad.qm_lr`,
- :ref:`bob.pad.face.resources.face_pad.qm_svm_aggregated_db`.

For a more detailed understanding of above pipe-lines you can also check corresponding configuration files:

- ``bob/pad/face/config/qm_one_class_gmm.py``,
- ``bob/pad/face/config/qm_one_class_svm_aggregated_db.py``,
- ``bob/pad/face/config/qm_lr.py``,
- ``bob/pad/face/config/qm_svm_aggregated_db.py``.

To run above algorithms on the :ref:`bob.pad.face.resources.databases.aggregated_db` database, using the ``photo-photo-video`` protocol, execute the following:

.. code-block:: sh

    $ spoof.py aggregated-db qm-one-class-gmm \
    --protocol photo-photo-video \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_1>

    $ spoof.py aggregated-db qm-one-class-svm-aggregated-db \
    --protocol photo-photo-video \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_2>

    $ spoof.py aggregated-db qm-lr \
    --protocol photo-photo-video \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_3>

    $ spoof.py aggregated-db qm-svm-aggregated-db \
    --protocol photo-photo-video \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_4>

.. tip::

    If you are in `idiap`_ you can use SGE grid to speed-up the calculations.
    Simply add ``--grid idiap`` argument to the above command. For example:


To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    ./bin/evaluate.py \
    --dev-files \
    <PATH_TO_STORE_THE_RESULTS_1>/photo-photo-video/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_2>/photo-photo-video/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_3>/photo-photo-video/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_4>/photo-photo-video/scores/scores-dev  \
    --eval-files \
    <PATH_TO_STORE_THE_RESULTS_1>/photo-photo-video/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_2>/photo-photo-video/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_3>/photo-photo-video/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_4>/photo-photo-video/scores/scores-eval \
    --legends \
    "IQM + one-class GMM + Aggregated Db" \
    "IQM + one-class SVM + Aggregated Db" \
    "IQM + two-class LR  + Aggregated Db" \
    "IQM + two-class SVM + Aggregated Db" \
    -F 7 \
    --criterion EER \
    --roc <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for the :ref:`bob.pad.face.resources.databases.aggregated_db` database are summarized in the Table below:

+------------------------+----------+----------+
|      Algorithm         |  EER,\%  |  HTER,\% |
+========================+==========+==========+
|   IQM + one-class GMM  |  22.075  |  14.470  |
+------------------------+----------+----------+
|   IQM + one-class SVM  |  35.537  |  24.317  |
+------------------------+----------+----------+
|   IQM + two-class LR   |  10.184  |  30.132  |
+------------------------+----------+----------+
|   IQM + two-class SVM  |  10.527  |  21.926  |
+------------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_anomaly_detection_aggr_db_ph_ph_vid.pdf>`

------------


Results for *video-video-photo* protocol
========================================================================

This section summarizes the evaluation results on the **video-video-photo** protocol of the Aggregated database for the following face PAD algorithms (for more details click on the corresponding algorithm):

- :ref:`bob.pad.face.resources.face_pad.qm_one_class_gmm`,
- :ref:`bob.pad.face.resources.face_pad.qm_one_class_svm_aggregated_db`,
- :ref:`bob.pad.face.resources.face_pad.qm_lr`,
- :ref:`bob.pad.face.resources.face_pad.qm_svm_aggregated_db`.

For a more detailed understanding of above pipe-lines you can also check corresponding configuration files:

- ``bob/pad/face/config/qm_one_class_gmm.py``,
- ``bob/pad/face/config/qm_one_class_svm_aggregated_db.py``,
- ``bob/pad/face/config/qm_lr.py``,
- ``bob/pad/face/config/qm_svm_aggregated_db.py``.

To run above algorithms on the :ref:`bob.pad.face.resources.databases.aggregated_db` database, using the ``video-video-photo`` protocol, execute the following:

.. code-block:: sh

    $ spoof.py aggregated-db qm-one-class-gmm \
    --protocol video-video-photo \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_1>

    $ spoof.py aggregated-db qm-one-class-svm-aggregated-db \
    --protocol video-video-photo \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_2>

    $ spoof.py aggregated-db qm-lr \
    --protocol video-video-photo \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_3>

    $ spoof.py aggregated-db qm-svm-aggregated-db \
    --protocol video-video-photo \
    --sub-directory <PATH_TO_STORE_THE_RESULTS_4>

.. tip::

    If you are in `idiap`_ you can use SGE grid to speed-up the calculations.
    Simply add ``--grid idiap`` argument to the above command. For example:


To evaluate the results computing EER, HTER and plotting ROC you can use the
following command:

.. code-block:: sh

    ./bin/evaluate.py \
    --dev-files \
    <PATH_TO_STORE_THE_RESULTS_1>/video-video-photo/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_2>/video-video-photo/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_3>/video-video-photo/scores/scores-dev  \
    <PATH_TO_STORE_THE_RESULTS_4>/video-video-photo/scores/scores-dev  \
    --eval-files \
    <PATH_TO_STORE_THE_RESULTS_1>/video-video-photo/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_2>/video-video-photo/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_3>/video-video-photo/scores/scores-eval \
    <PATH_TO_STORE_THE_RESULTS_4>/video-video-photo/scores/scores-eval \
    --legends \
    "IQM + one-class GMM + Aggregated Db" \
    "IQM + one-class SVM + Aggregated Db" \
    "IQM + two-class LR  + Aggregated Db" \
    "IQM + two-class SVM + Aggregated Db" \
    -F 7 \
    --criterion EER \
    --roc <PATH_TO_STORE_THE_RESULTS>/ROC.pdf

The EER/HTER errors for the :ref:`bob.pad.face.resources.databases.aggregated_db` database are summarized in the Table below:

+------------------------+----------+----------+
|      Algorithm         |  EER,\%  |  HTER,\% |
+========================+==========+==========+
|   IQM + one-class GMM  |  13.503  |  29.794  |
+------------------------+----------+----------+
|   IQM + one-class SVM  |  18.234  |  39.502  |
+------------------------+----------+----------+
|   IQM + two-class LR   |  1.499   |  30.268  |
+------------------------+----------+----------+
|   IQM + two-class SVM  |  1.422   |  24.901  |
+------------------------+----------+----------+

The ROC curves for the particular experiment can be downloaded from here:

:download:`ROC curve <img/ROC_iqm_anomaly_detection_aggr_db_vid_vid_ph.pdf>`

------------


.. include:: links.rst


