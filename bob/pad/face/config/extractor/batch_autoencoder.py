#!/usr/bin/env python

from bob.pad.face.extractor import BatchAutoencoder

#=======================================================================================
# Define instances here:

CODE_LAYER_FEATURES_FLAG = True

autoencoder_code_layer = BatchAutoencoder(code_layer_features_flag = CODE_LAYER_FEATURES_FLAG)


#=======================================================================================
CODE_LAYER_FEATURES_FLAG = False

autoencoder_mse = BatchAutoencoder(code_layer_features_flag = CODE_LAYER_FEATURES_FLAG)