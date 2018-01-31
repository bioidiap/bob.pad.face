#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Olegs Nikisins
"""
#==============================================================================
# Import here:

from torchvision import transforms

from bob.pad.face.database import CELEBAPadDatabase

from torch import nn


#==============================================================================
# Define parameters here:

"""
Note: do not change names of the below constants.
"""
NUM_EPOCHS = 100 # Maximum number of epochs
BATCH_SIZE =  128 # Size of the batch
LEARNING_RATE = 1e-3 # Learning rate


"""
Transformations to be applied sequentially to the input PIL image.
Note: the variable name ``transform`` must be the same in all configuration files.
"""
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


"""
Set the parameters of the DataFolder dataset class.
Note: do not change the name ``kwargs``.
"""
bob_hldi_instance = CELEBAPadDatabase(original_directory = "", original_extension = "")

kwargs = {}
kwargs["data_folder"] = "NO NEED TO SET HERE, WILL BE SET IN THE TRAINING SCRIPT"
kwargs["transform"] = transform
kwargs["extension"] = '.hdf5'
kwargs["bob_hldi_instance"] = bob_hldi_instance
kwargs["hldi_type"] = "pad"
kwargs["groups"] = ['train']
kwargs["protocol"] = 'grandtest'
kwargs["purposes"] = ['real']
kwargs["allow_missing_files"] = True


"""
Define the network to be trained as a class, named ``Network``.
Note: Do not change the name of the below class.
"""

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)

        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

"""
Define the loss to be used for training.
Note: do not change the name of the below variable.
"""
loss_type = nn.MSELoss()
