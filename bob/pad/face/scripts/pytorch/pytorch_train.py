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

    parser.add_argument("data_folder", type=str,
                        help="A directory containing the training data.")

    parser.add_argument("save_folder", type=str,
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
    """
    TODO: regactor this function
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x


#==============================================================================
def process_verbosity(verbosity,
                      epoch,
                      num_epochs,
                      loss_value,
                      epoch_step,
                      batch_tensor,
                      save_folder):
    """
    Report results based on the verbose level.

    1. If verbosity level is 1: loss is printed for each epoch.

    2. If verbosity levle is greater than 1: both loss is printed and
       a reconstructed image is saved efter each ``epoch_step`` epochs.

    **Parameters:**

    ``verbosity``: py:class:`int`
        Verbosity level.

        TODO: add documentation
    """

    if verbosity > 0:

        print ('epoch [{}/{}], loss:{:.4f}'.format(epoch, num_epochs, loss_value))

        if verbosity > 1:

            if epoch % epoch_step == 0:

                pic = to_img(batch_tensor)
                save_image( pic, os.path.join(save_folder, 'image_{}.png'.format(epoch)) )


#==============================================================================
def main(cmd_params=None):
    """
    TODO: add documentation
    """

    epoch_step = 10 # save images and trained model after each ``epoch_step`` epoch

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

        batch_num = 0

        for data in dataloader:

            batch_num = batch_num + 1

            img, _ = data

            img = Variable(img)
            #===================forward========================================
            output = model(img)
            loss = loss_type(output, img)
            #===================backward=======================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_num == len(dataloader) - 1: # process verbosity using penultimate batch, because the
            # last batch can be smaller than BATCH_SIZE.

                process_verbosity(verbosity = verbosity,
                                  epoch = epoch+1,
                                  num_epochs = config_module.NUM_EPOCHS,
                                  loss_value = loss.data[0],
                                  epoch_step = epoch_step,
                                  batch_tensor = output.cpu().data,
                                  save_folder = save_folder)

        if (epoch+1) % epoch_step == 0:

            torch.save(model.state_dict(), os.path.join(save_folder, 'model_{}.pth'.format(epoch+1)))











