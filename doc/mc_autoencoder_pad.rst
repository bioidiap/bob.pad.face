

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

    ./bin/spoof.py \                                        # spoof.py is used to run the preprocessor
    celeb-a \                                               # run for CelebA database
    lbp-svm \                                               # required by spoof.py, but unused
    --skip-extractor-training --skip-extraction --skip-projector-training --skip-projection --skip-score-computation --allow-missing-files \    # execute only preprocessing step
    --grid idiap \                                          # use grid, only for Idiap users, remove otherwise
    --groups train \                                        # preprocess only training set of CelebA
    --preprocessor rgb-face-detect-check-quality-128x128 \  # preprocessor entry point
    --sub-directory <PATH_TO_STORE_THE_RESULTS>             # define your path here

Running above command, the RGB facial images are aligned and cropped from the training set of the CelebA database. Additionally, a quality assessment is applied to each facial image.
More specifically, an eye detection algorithm is applied to face images, assuring the deviation of eye coordinates from expected positions is not significant.
See [NGM19]_ for more details.

Once above script is completed, the data suitable for autoencoder training is located in the folder ``<PATH_TO_STORE_THE_RESULTS>/preprocessed/``. Now the autoencoder can be trained.
The training procedure is explained in the **Convolutional autoencoder** section in the documentation of the ``bob.learn.pytorch`` package.

.. note::

  Functionality of ``bob.pad.face`` is used to compute the training data. Install and follow the documentation of ``bob.learn.pytorch`` to train the autoencoders. This functional decoupling helps to avoid the dependency of
  ``bob.pad.face`` from **PyTorch**.


.. include:: links.rst


2. Fine-tune N AEs on multi-channel data from WMCA (legacy name BATL) database
=================================================================================

Following the training procedure of [NGM19]_, the autoencoders are next fine-tuned on the multi-channel (**MC**) data from WMCA.
In this example, MC training data is a stack of gray-scale, NIR, and Depth (BW-NIR-D) facial images.

To prepare the training data one can use the following command:


.. code-block:: sh

    ./bin/spoof.py \                                                    # spoof.py is used to run the preprocessor
    batl-db-rgb-ir-d-grandtest \                                        # WMCA database instance allowing to load RGB-NIR-D channels
    lbp-svm \                                                           # required by spoof.py, but unused
    --skip-extractor-training --skip-extraction --skip-projector-training --skip-projection --skip-score-computation --allow-missing-files \    # execute only preprocessing step
    --grid idiap \                                                      # use grid, only for Idiap users, remove otherwise
    --preprocessor video-face-crop-align-bw-ir-d-channels-3x128x128 \   # preprocessor entry point
    --sub-directory <PATH_TO_STORE_THE_RESULTS>                         # define your path here

Once above script is completed, the MC data suitable for autoencoder fine-tuning is located in the folder ``<PATH_TO_STORE_THE_RESULTS>/preprocessed/``.
Now the autoencoder can be fine-tuned. Again, the fine-tuning procedure is explained in the **Convolutional autoencoder** section in the documentation of the ``bob.learn.pytorch`` package.


3. Train an MLP using multi-channel autoencoder latent embeddings from WMCA
=================================================================================

Once auto-encoders are pre-trained and fine-tuned, the latent embeddings can be computed passing the multi-channel (MC) BW-NIR-D images from the WMCA database through the encoder, see [NGM19]_ for more details. These latent embeddings (feature vectors) are next used to train an MLP classifying input MC samples into bona-fide or attack classes.

The first step to be done is the registration of an extractor computing latent embeddings. To do so, a file defining an instance of **MultiNetPatchExtractor** class must be created:

.. code-block:: sh

    from bob.ip.pytorch_extractor import MultiNetPatchExtractor
    from bob.bio.video.utils import FrameSelector
    from bob.bio.video.extractor import Wrapper
    from torchvision import transforms
    from bob.learn.pytorch.architectures import ConvAutoencoder

    # transform to be applied to input patches:
    TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    # use latent embeddings in the feature extractor:
    NETWORK_AE = ConvAutoencoder(return_latent_embedding = True)

    # use specific/unique model for each patch. Models pre-trained on CelebA and fine-tuned on BATL:
    MODEL_FILE = ["SUBSTITUTE_THE_PATH_TO_PRETRAINED_AE_MODEL"]

    PATCHES_NUM = [0] # patches to be used in the feature extraction process
    PATCH_RESHAPE_PARAMETERS = [3, 128, 128]  # reshape vectorized patches to this dimensions before passing to the Network

    _image_extractor = MultiNetPatchExtractor(transform = TRANSFORM,
                                              network = NETWORK_AE,
                                              model_file = MODEL_FILE,
                                              patches_num = PATCHES_NUM,
                                              patch_reshape_parameters = PATCH_RESHAPE_PARAMETERS,
                                              color_input_flag = True,
                                              urls = None,
                                              archive_extension = '.tar.gz')

    extractor = Wrapper(extractor = _image_extractor,
                        frame_selector = FrameSelector(selection_style = "all"))

Suppose, above configuration file is located in ``bob.pad.face`` package in the following location: ``bob/pad/face/config/extractor/multi_net_patch_extractor.py``. Then it can be registered in ``setup.py`` by adding the following string to the list of registered extractors ``bob.pad.extractor``:

.. code-block:: sh

    'multi-net-patch-extractor = bob.pad.face.config.extractor.multi_net_patch_extractor:extractor',


Once an extractor is registered, to compute the latent embeddings (encoder output) the following command can be used:

.. code-block:: sh

    ./bin/spoof.py \                                                        # spoof.py is used to extract embeddings
    batl-db-rgb-ir-d-grandtest \                                            # WMCA database instance allowing to load RGB-NIR-D channels
    lbp-svm \                                                               # required by spoof.py, but unused
    --preprocessor video-face-crop-align-bw-ir-d-channels-3x128x128-vect \  # entry point defining preprocessor
    --extractor multi-net-patch-extractor \                                 # entry point defining extractor
    --skip-projector-training --skip-projection --skip-score-computation --allow-missing-files \  # execute preprocessing and extraction only
    --grid idiap \                                                          # use grid, for Idiap users only, remove otherwise
    --sub-directory <PATH_TO_STORE_THE_RESULTS>                             # define your path here

.. note::

  Make sure the ``bob.learn.pytorch`` and ``bob.ip.pytorch_extractor`` packages are installed before running above command.

Once above script is completed, the MC latent encodings to be used for MLP training are located in the folder ``<PATH_TO_STORE_THE_RESULTS>/extracted/``.
Again, the training procedure is explained in the **MLP** section in the documentation of the ``bob.learn.pytorch`` package.

