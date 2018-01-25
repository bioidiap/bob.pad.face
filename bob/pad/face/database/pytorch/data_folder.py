#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

import torch.utils.data as data

import os

from bob.bio.video.utils import FrameContainer

import bob.io.base

import random
random.seed( a = 7 )

import PIL

import numpy as np

#==============================================================================
def get_file_names_and_labels(files, data_folder, extension = ".hdf5", hldi_type = "pad"):
    """
    Get absolute names of the corresponding file objects and their class labels.

    **Parameters:**

    ``files`` : [File]
        A list of files objects defined in the High Level Database Interface
        of the particular datbase.

    ``data_folder`` : str
        A directory containing the training data.

    ``extension`` : str
        Extension of the data files. Default: ".hdf5" .

    ``hldi_type`` : str
        Type of the high level database interface. Default: "pad".
        Note: this is the only type supported at the moment.

    **Returns:**

    ``file_names_and_labels`` : [(str, int)]
        A list of tuples, where each tuple contain an absolute filename and
        a corresponding label of the class.
    """

    file_names_and_labels = []

    if hldi_type == "pad":

        for f in files:

            if f.attack_type is None:

                label = 0

            else:

                label = 1

            file_names_and_labels.append( ( os.path.join(data_folder, f.path + extension), label ) )

    return file_names_and_labels


#==============================================================================
class DataFolder(data.Dataset):
    """
    A generic data loader compatible with Bob High Level Database Interfaces
    (HLDI). Only HLDI's of bob.pad.face are currently supported.
    """

    def __init__(self, data_folder,
                 transform = None,
                 extension = '.hdf5',
                 bob_hldi_instance = None,
                 hldi_type = "pad",
                 groups = ['train', 'dev', 'eval'],
                 protocol = 'grandtest',
                 purposes=['real', 'attack'],
                 allow_missing_files = True,
                 **kwargs):
        """
        **Parameters:**

        ``data_folder`` : str
            A directory containing the training data.

        ``transform`` : callable
            A function/transform that  takes in a PIL image, and returns a
            transformed version. E.g, ``transforms.RandomCrop``. Default: None.

        ``extension`` : str
            Extension of the data files. Default: ".hdf5".
            Note: this is the only extension supported at the moment.

        ``bob_hldi_instance`` : object
            An instance of the HLDI interface. Only HLDI's of bob.pad.face
            are currently supported.

        ``hldi_type`` : str
            String defining the type of the HLDI. Default: "pad".
            Note: this is the only option currently supported.

        ``groups`` : str or [str]
            The groups for which the clients should be returned.
            Usually, groups are one or more elements of ['train', 'dev', 'eval'].
            Default: ['train', 'dev', 'eval'].

        ``protocol`` : str
            The protocol for which the clients should be retrieved.
            Default: 'grandtest'.

        ``purposes`` : str or [str]
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.
            Default: ['real', 'attack'].

        ``allow_missing_files`` : str or [str]
            The missing files in the ``data_folder`` will not break the
            execution if set to True.
            Default: True.
        """

        self.data_folder = data_folder
        self.transform = transform
        self.extension = extension
        self.bob_hldi_instance = bob_hldi_instance
        self.hldi_type = hldi_type
        self.groups = groups
        self.protocol = protocol
        self.purposes = purposes
        self.allow_missing_files = allow_missing_files

        if bob_hldi_instance is not None:

            files = bob_hldi_instance.objects(groups = self.groups,
                                              protocol = self.protocol,
                                              purposes = self.purposes,
                                              **kwargs)

            file_names_and_labels = get_file_names_and_labels(files = files,
                                        data_folder = self.data_folder,
                                        extension = self.extension,
                                        hldi_type = self.hldi_type)

            if self.allow_missing_files: # return only existing files

                file_names_and_labels = [f for f in file_names_and_labels if os.path.isfile(f[0])]

        else:

            # TODO - add behaviour similar to image folder
            file_names_and_labels = []

        self.file_names_and_labels = file_names_and_labels


    #==========================================================================
    def __getitem__(self, index):
        """
        Returns an image, possibly transformed, and a target class given index.

        **Parameters:**

        ``index`` : int.
            An index of the sample to return.

        **Returns:**

        ``pil_img`` : Tensor or PIL Image
            If ``self.transform`` is defined the output is the torch.Tensor,
            otherwise the output is an instance of the PIL.Image.Image class.

        ``target`` : int
            Index of the class.
        """

        path, target = self.file_names_and_labels[index]

        video = FrameContainer(bob.io.base.HDF5File(path))

        fn = random.randint(0, len(video))

        img_array = video[fn][1] # The size now is (3 x W x H)

        img_array_tr = np.swapaxes(img_array, 1, 2)
        img_array_tr = np.swapaxes(img_array_tr, 0, 2)

        pil_img = PIL.Image.fromarray( img_array_tr ) # convert to PIL and from array of size (H x W x 3)

        if self.transform is not None:

            pil_img = self.transform(pil_img)

        return pil_img, target


    #==========================================================================
    def __len__(self):
        """
        **Returns:**

        ``len`` : int
            The length of the file list.
        """
        return len(self.file_names_and_labels)




