#!/usr/bin/env python

from bob.pad.face.extractor import BatchAutoencoder

#=======================================================================================
# Define instances here:

CODE_LAYER_FEATURES_FLAG = True

autoencoder_code_layer = BatchAutoencoder(model_file = "",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")


#=======================================================================================
CODE_LAYER_FEATURES_FLAG = False

autoencoder_mse = BatchAutoencoder(model_file = "",
                                   code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                   config_file = "autoencoder/autoencoder_config.py",
                                   config_group = "bob.pad.face.config.pytorch")