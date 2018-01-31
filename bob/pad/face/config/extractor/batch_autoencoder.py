#!/usr/bin/env python

from bob.pad.face.extractor import BatchAutoencoder

#=======================================================================================
# Define instances here:

CODE_LAYER_FEATURES_FLAG = True

autoencoder_code_layer = BatchAutoencoder(model_file = "",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_celeba_100_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/celeba_pretraining/model_100.pth",
                                                            code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                                            config_file = "autoencoder/autoencoder_config_celeba.py",
                                                            config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_aggr_db_1_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/aggregated_db_domain_adaptation/model_1.pth",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_aggr_db_10_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/aggregated_db_domain_adaptation/model_10.pth",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_aggr_db_20_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/aggregated_db_domain_adaptation/model_20.pth",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_aggr_db_30_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/aggregated_db_domain_adaptation/model_30.pth",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_aggr_db_40_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/aggregated_db_domain_adaptation/model_40.pth",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_aggr_db_50_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/aggregated_db_domain_adaptation/model_50.pth",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")

autoencoder_code_layer_aggr_db_100_epochs = BatchAutoencoder(model_file = "/idiap/temp/onikisins/project/ODIN/experiment_data/pytorch_experiments/domain_adaptation/aggregated_db_domain_adaptation/model_100.pth",
                                          code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                          config_file = "autoencoder/autoencoder_config.py",
                                          config_group = "bob.pad.face.config.pytorch")


#=======================================================================================
CODE_LAYER_FEATURES_FLAG = False

autoencoder_mse = BatchAutoencoder(model_file = "",
                                   code_layer_features_flag = CODE_LAYER_FEATURES_FLAG,
                                   config_file = "autoencoder/autoencoder_config.py",
                                   config_group = "bob.pad.face.config.pytorch")