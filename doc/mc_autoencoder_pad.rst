

.. _bob.pad.face.mc_autoencoder_pad:


=============================================
 Multi-channel face PAD using autoencoders
=============================================

This section explains how to run a complete face PAD experiment using multi-channel autoencoder-based face PAD system, as well as a training work-flow.

The system discussed in this section is introduced the following publication [NGM19]_. It is **strongly recommended** to check the publication for better understanding
of the described work-flow.

.. warning::

   Algorithms introduced in this section might be in the process of publishing. Therefore, it is not
   allowed to publish results introduced in this section without permission of the owner of the package.
   If you are planning to use the results from this section, please contact the owner of the package first.
   Please check the ``setup.py`` for contact information.


Running face PAD Experiments
------------------------------

Please refer to :ref:`bob.pad.face.baselines` section of current documentation for more details on how to run the face PAD experiments and setup the databases.


Training multi-channel autoencoder-based face PAD system.
----------------------------------------------------------------

As introduced in the paper [NGM19]_, the training of the system is composed of three main steps, which are summarize in the following table:

+----------------------+----------------------+---------------------+
| Train step           | Training data        | DB, classes used    |
+----------------------+----------------------+---------------------+
| Train N AEs          | RGB face regions     | CelebA, BF          |
+----------------------+----------------------+---------------------+
| Fine-tune N AEs      | MC face regions      | WMCA, BF            |
+----------------------+----------------------+---------------------+
| Train an MLP         | MC latent encodings  | WMCA, BF and PA     |
+----------------------+----------------------+---------------------+

In the above table, **BF** and **PA** stands for samples from **bona-fide** and **presentation attack** classes.

As one can conclude from the table, CelebA and WMCA databases must be installed before the training can take place.
See :ref:`bob.pad.face.baselines` for databases installation details.


1. Train N AEs on RGB data from CelebA
===========================================

In [NGM19]_ N autoencoders are trained, one for each facial region, here for explanatory purposes, a system containing **one** autoencoder is observed, thus N=1.
This autoencoder is first pre-trained using RGB images of entire face, which are cropped from CelebA database.

To prepare the training data one can use the following command:


.. code-block:: sh

    ./bin/spoof.py \    # spoof.py is used to run the preprocessor
    celeb-a \   # run for CelebA database
    lbp-svm \   # required by spoof.py, but unused
    --skip-extractor-training --skip-extraction --skip-projector-training --skip-projection --skip-score-computation --allow-missing-files \    # execute only preprocessing step
    --grid idiap \    # use grid, only for Idiap users, remove otherwise
    --groups train \    # preprocess only training set of CelebA
    --preprocessor rgb-face-detect-check-quality-128x128 \    # preprocessor entry point
    --sub-directory <PATH_TO_STORE_THE_RESULTS>   # define your path here

Running above command, the RGB facial images are aligned and cropped from the training set of the CelebA database. Additionally, a quality assessment is applied to each facial image.
More specifically, an eye detection algorithm is applied to face images, assuring the deviation of eye coordinates from expected positions is not significant.
See [NGM19]_ for more details.


.. include:: links.rst





