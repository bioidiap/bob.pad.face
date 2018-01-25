#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Olegs Nikisins
"""


"""
The following steps are performed in this code:

1. ????

2. ????
"""
#==============================================================================
# Import here:

import argparse
import importlib
import os

from bob.pad.face.database.pytorch import DataFolder

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image


#import torchvision
#from torch import nn
#from torch.autograd import Variable
#from torchvision import transforms
#from torchvision.utils import save_image
#from torchvision.datasets import MNIST
#import os


#==============================================================================
def parse_arguments(cmd_params=None):
    """
    Parse command line arguments.

    **Parameters:**

    ``cmd_params``: []
        An optional list of command line arguments. Default: None.

    **Returns:**

    ``data_folder``: py:class:`string`
        A directory containing the training data.

    ``save_folder``: py:class:`string`
        A directory to save the results of training to.

    ``relative_mod_name``: py:class:`string`
        Relative name of the module to import configurations from.

    ``config_group``: py:class:`string`
        Group/package name containing the configuration file.

    ``verbosity``: py:class:`int`
        Verbosity level.
    """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("data-folder", type=str,
                        help="A directory containing the training data.")

    parser.add_argument("save-folder", type=str,
                        help="A directory to save the results of training to.")

    parser.add_argument("-c", "--config-file", type=str, help="Relative name of the config file defining "
                        "the network, training data, and training parameters.",
                        default = "autoencoder/autoencoder_config.py")

    parser.add_argument("-cg", "--config-group", type=str, help="Name of the group, where config file is stored.",
                        default = "bob.pad.face.config.pytorch")

    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="Increase output verbosity.")

    if cmd_params is not None:
        args = parser.parse_args(cmd_params)
    else:
        args = parser.parse_args()

    data_folder = args.data_folder
    save_folder = args.save_folder

    config_file = args.config_file
    config_group = args.config_group
    verbosity = args.verbosity

    relative_mod_name = '.' + os.path.splitext(config_file)[0].replace(os.path.sep, '.')

    return data_folder, save_folder, relative_mod_name, config_group, verbosity


#==============================================================================
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x


#==============================================================================
def main(cmd_params=None):
    """

    """

    data_folder, save_folder, relative_mod_name, config_group, verbosity = \
                                parse_arguments(cmd_params = cmd_params)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    config_module = importlib.import_module(relative_mod_name, config_group)

    dataset_kwargs = config_module.kwargs

    dataset_kwargs["data_folder"] = data_folder

    dataset = DataFolder(**dataset_kwargs)

    dataloader = DataLoader(dataset,
                            batch_size = config_module.BATCH_SIZE,
                            shuffle = True)

    model = config_module.Network()

    loss_type = config_module.loss_type

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config_module.LEARNING_RATE,
                                 weight_decay=1e-5)

    for epoch in range(config_module.NUM_EPOCHS):

        for data in dataloader:

            img, _ = data
            img = Variable(img)
            #===================forward========================================
            output = model(img)
            loss = loss_type(output, img)
            #===================backward=======================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #===================log================================================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, config_module.NUM_EPOCHS, loss.data[0]))
    #    save_name = './conv_autoencoder'+str(epoch)+'.pth'
    #    torch.save(model.state_dict(), save_name)
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image( pic, os.path.join(save_folder, 'image_{}.png'.format(epoch)) )




#    torch.save(model.state_dict(), os.path.join(save_folder, './conv_autoencoder.pth'))




#transform = transforms.Compose([transforms.Resize((64, 64)),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                ])
#
#
#dataset_kwargs = {}
#dataset_kwargs["data_folder"] = "WILL BE SET IN THE TRAINING SCRIPT"
#dataset_kwargs["transform"] = transform
#dataset_kwargs["extension"] = '.hdf5'
#dataset_kwargs["bob_hldi_instance"] = bob_hldi_instance
#dataset_kwargs["hldi_type"] = "pad"
#dataset_kwargs["groups"] = ['train']
#dataset_kwargs["protocol"] = 'grandtest'
#dataset_kwargs["purposes"] = ['real']
#dataset_kwargs["allow_missing_files"] = True
#
#
#all_data = importlib.import_module(relative_mod_name, config_group)
#
#data_folder = '/idiap/temp/onikisins/project/ODIN/experiment_data/face_detect_align_experiments/aggregated_db/experiment_3/preprocessed/'































