#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 2017

@author: Olegs Nikisins
"""

#==============================================================================
# Import what is needed here:

from bob.bio.base.preprocessor import Preprocessor

import bob.bio.video

import numpy as np

import random
random.seed(7)


#==============================================================================
class VideoSparseCoding(Preprocessor, object):
    """
    This class is designed to compute "feature vectors" for all stacks of facial
    images using sparse coding. The feature vector is computed for each stack
    containing ``block_length`` images.

    The maximum number of facial stacks per video is:
    (``num_of_frames_in_video`` - ``block_length``).
    However, the number of facial volumes might be less than above, because
    frames with small faces ( < min_face_size ) are discarded.

    The feature vector is computed as follows............

    **Parameters:**

    ``block_size`` : :py:class:`int`
        The spatial size of facial patches. Default: 5 .

    ``block_length`` : :py:class:`int`
        The temporal length of the stack of facial images / number of frames
        per stack. Default: 10 .

    ``min_face_size`` : :py:class:`int`
        Discard frames with face of the size less than ``min_face_size``.
        Default: 50 .

    ``norm_face_size`` : :py:class:`int`
        The size of the face after normalization. Default: 64 .
    """


    #==========================================================================
    def __init__(self,
                 block_size = 5,
                 block_length = 10,
                 min_face_size = 50,
                 norm_face_size = 64,
                 **kwargs):

        super(VideoSparseCoding, self).__init__(block_size = block_size,
                                                block_length = block_length,
                                                min_face_size = min_face_size,
                                                norm_face_size = norm_face_size)

        self.block_size = block_size
        self.block_length = block_length
        self.min_face_size = min_face_size
        self.norm_face_size = norm_face_size

        self.video_preprocessor = bob.bio.video.preprocessor.Wrapper()


    #==========================================================================
    def crop_norm_face_grayscale(self, image, annotations, face_size):
        """
        This function crops the face in the input Gray-scale image given
        annotations defining the face bounding box. The size of the face is
        also normalized to the pre-defined dimensions.

        The algorithm is identical to the following paper:
        "On the Effectiveness of Local Binary Patterns in Face Anti-spoofing"

        **Parameters:**

        ``image`` : 2D :py:class:`numpy.ndarray`
            Gray-scale input image.

        ``annotations`` : :py:class:`dict`
            A dictionary containing annotations of the face bounding box.
            Dictionary must be as follows:
            ``{'topleft': (row, col), 'bottomright': (row, col)}``

        ``face_size`` : :py:class:`int`
            The size of the face after normalization.

        **Returns:**

        ``normbbx`` : 2D :py:class:`numpy.ndarray`
            Cropped facial image of the size (self.face_size, self.face_size).
        """

        cutframe = image[annotations['topleft'][0]:annotations['bottomright'][0],
                 annotations['topleft'][1]:annotations['bottomright'][1]]

        tempbbx = np.ndarray((face_size, face_size), 'float64')
        normbbx = np.ndarray((face_size, face_size), 'uint8')
        bob.ip.base.scale(cutframe, tempbbx) # normalization
        tempbbx_ = tempbbx + 0.5
        tempbbx_ = np.floor(tempbbx_)
        normbbx = np.cast['uint8'](tempbbx_)

        return normbbx


    #==========================================================================
    def crop_norm_faces_grayscale(self, images, annotations, face_size):
        """
        This function crops and normalizes faces in a stack of images given
        annotations of the face bounding box for the first image in the stack.

        **Parameters:**

        ``images`` : 3D :py:class:`numpy.ndarray`
            A stack of gray-scale input images. The size of the array is
            (n_images x n_rows x n_cols).

        ``annotations`` : :py:class:`dict`
            A dictionary containing annotations of the face bounding box.
            Dictionary must be as follows:
            ``{'topleft': (row, col), 'bottomright': (row, col)}``

        ``face_size`` : :py:class:`int`
            The size of the face after normalization.

        **Returns:**

        ``normbbx`` : 3D :py:class:`numpy.ndarray`
            A stack of normalized faces.
        """

        normbbx = []

        for image in images:

            normbbx.append( self.crop_norm_face_grayscale(image, annotations, face_size) )

        normbbx = np.stack(normbbx)

        return normbbx


    #==========================================================================
    def select_all_blocks(self, images, block_size):
        """
        Extract all possible 3D blocks from a stack of images.

        ``images`` : 3D :py:class:`numpy.ndarray`
            A stack of gray-scale input images. The size of the array is
            (``n_images`` x ``n_rows`` x ``n_cols``).

        ``block_size`` : :py:class:`int`
            The spatial size of patches. The size of extracted 3D blocks is:
            (``n_images`` x ``block_size`` x ``block_size``).
        """

        (_, row_num, col_num) = images.shape

        all_blocks = []

        for row in range(row_num - block_size):

            for col in range(col_num - block_size):

                block = images[:, row:row+block_size, col:col+block_size]

                all_blocks.append( block )

        return all_blocks


    #==========================================================================
    def convert_frame_cont_to_grayscale_array(self, frame_cont):
        """
        Convert color video stored in the frame container into 3D array storing
        gray-scale frames. The dimensions of the output array are:
        (n_frames x n_rows x n_cols).

        **Parameters:**

        ``frames`` : FrameContainer
            Video data stored in the FrameContainer, see
            ``bob.bio.video.utils.FrameContainer`` for further details.

        **Returns:**

        ``result_array`` : 3D :py:class:`numpy.ndarray`
            A stack of gray-scale frames. The size of the array is
            (n_frames x n_rows x n_cols).
        """

        result_array = []

        for frame in frame_cont:

            image = frame[1]

            result_array.append( bob.ip.color.rgb_to_gray(image) )

        result_array = np.stack(result_array)

        return result_array


    #==========================================================================
    def get_all_blocks_from_color_channel(self, video, annotations, block_size, block_length, min_face_size, norm_face_size):
        """
        Extract all 3D blocks from facial region of the input 3D array.
        Input 3D array represents one color channel of the video or a gray-scale
        video. Blocks are extracted from all 3D facial volumes. Facial volumes
        overlap with a shift of one frame.

        The size of the facial volume is:
        (``block_length`` x ``norm_face_size`` x ``norm_face_size``).

        The maximum number of available facial volumes in the video:
        (``num_of_frames_in_video`` - ``block_length``).
        However the final number of facial volumes might be less than above,
        because frames with small faces ( < min_face_size ) are discarded.

        **Parameters:**

        ``video`` : 3D :py:class:`numpy.ndarray`
            A stack of gray-scale input images. The size of the array is
            (n_images x n_rows x n_cols).

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure:
            ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``,
            where
            ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding
            box in frame N.

        ``block_size`` : :py:class:`int`
            The spatial size of facial patches.

        ``block_length`` : :py:class:`int`
            The temporal length of the stack of facial images / number of frames
            per stack.

        ``min_face_size`` : :py:class:`int`
            Discard frames with face of the size less than ``min_face_size``.

        ``norm_face_size`` : :py:class:`int`
            The size of the face after normalization.

        **Returns:**

        ``all_blocks`` : [[3D :py:class:`numpy.ndarray`]]
            Internal list contains all possible 3D blocks/volumes extracted from
            a particular stack of facial images. The dimensions of each 3D block:
            (block_length x block_size x block_size).
            The number of possible blocks is: (norm_face_size - block_size)^2.

            The length of the outer list is equal to the number of possible
            facial stacks in the input video:
            (``num_of_frames_in_video`` - ``block_length``).
            However, the final number of facial volumes might be less than above,
            because frames with small faces ( < min_face_size ) are discarded.
        """

        annotated_frames = annotations.keys()

        all_blocks = []

        for fn in range(len(video)-block_length):

            if str(fn) in annotated_frames: # process if frame is annotated

                frame_annotations = annotations[str(fn)]

                face_size = np.min(np.array(frame_annotations['bottomright']) - np.array(frame_annotations['topleft']))

                if face_size >= min_face_size: # process is face is large enough

                    # Selected 3D stacks of images. Stack has ``block_length`` images.
                    stack_of_images = video[fn:fn + block_length, :, :]

                    # 3D stacks of normalized face images.
                    faces = self.crop_norm_faces_grayscale(stack_of_images, frame_annotations, norm_face_size)

                    # A list with all blocks per stack of facial images.
                    list_all_blocks_per_stack = self.select_all_blocks(faces, block_size)

                    all_blocks.append( list_all_blocks_per_stack )

        return all_blocks


    #==========================================================================
    def extract_patches_from_blocks(self, all_blocks):
        """
        Extract frontal, central-horizontal and central-vertical patches from
        all blocks returned by ``get_all_blocks_from_color_channel``
        method of this class. The patches are returned in a vectorized form.

        **Parameters:**

        ``all_blocks`` : [[3D :py:class:`numpy.ndarray`]]
            Internal list contains all possible 3D blocks/volumes extracted from
            a particular stack of facial images. The dimensions of each 3D block:
            (block_length x block_size x block_size).
            The number of possible blocks is: (norm_face_size - block_size)^2.

            The length of the outer list is equal to the number of possible
            facial stacks in the input video:
            (``num_of_frames_in_video`` - ``block_length``).
            However, the final number of facial volumes might be less than above,
            because frames with small faces ( < min_face_size ) are discarded.

        **Returns:**

        ``frontal_patches`` : [2D :py:class:`numpy.ndarray`]
            Each element in the list contains an array of vectorized frontal
            patches for the particular stack of facial images.
            The size of each array is:
            ( (``norm_face_size`` - ``block_size``)^2 x ``block_size``^2 ).
            The maximum length of the list is:
            (``num_of_frames_in_video`` - ``block_length``)

        ``horizontal_patches`` : [2D :py:class:`numpy.ndarray`]
            Each element in the list contains an array of vectorized horizontal
            patches for the particular stack of facial images.
            The size of each array is:
            ( (``norm_face_size`` - ``block_size``)^2 x ``block_length``*``block_size`` ).
            The maximum length of the list is:
            (``num_of_frames_in_video`` - ``block_length``)

        ``vertical_patches`` : [2D :py:class:`numpy.ndarray`]
            Each element in the list contains an array of vectorized vertical
            patches for the particular stack of facial images.
            The size of each array is:
            ( (``norm_face_size`` - ``block_size``)^2 x ``block_length``*``block_size`` ).
            The maximum length of the list is:
            (``num_of_frames_in_video`` - ``block_length``)
        """

        lenghth, row_num, col_num = all_blocks[0][0].shape

        selected_row = np.int(row_num/2)

        selected_col = np.int(col_num/2)

        frontal_patches = []
        horizontal_patches = []
        vertical_patches = []

        # volume - is a list of 3D blocks for a particular stack of facial images.
        for volume in all_blocks:

            volume_frontal_patches = []
            volume_horizontal_patches = []
            volume_vertical_patches = []

            for block in volume:

                frontal_patch = block[0, :, :] # the frontal patch of a block. Size: (row_num x col_num)
                volume_frontal_patches.append(frontal_patch.flatten())

                horizontal_patch = block[:, selected_row, :] # the central-horizontal patch of a block. Size: (lenghth x col_num), where
                # lenghth = block_length, col_num = block_size.
                volume_horizontal_patches.append(horizontal_patch.flatten())

                vertical_patch = block[:, :, selected_col] # the central-vertical patch of a block. Size: (lenghth x row_num)
                volume_vertical_patches.append(vertical_patch.flatten())

            frontal_patches.append( np.stack(volume_frontal_patches) )

            horizontal_patches.append( np.stack(volume_horizontal_patches) )

            vertical_patches.append( np.stack(volume_vertical_patches) )

        return frontal_patches, horizontal_patches, vertical_patches


    #==========================================================================
    def __select_random_patches_single_list(self, patches, n_patches):
        """
        This method is called by ``select_random_patches`` method to process
        all lists of patches.

        **Parameters:**

        ``patches`` : [2D :py:class:`numpy.ndarray`]
            Each element in the list contains an array of vectorized
            patches for the particular stack of facial images.
            The size of each array is:
            ( (``norm_face_size`` - ``block_size``)^2 x ``block_size``^2 ).
            The maximum length of the list is:
            (``num_of_frames_in_video`` - ``block_length``)

        ``n_patches`` : :py:class:`int`
            Number of randomly selected patches.

        **Returns:**

        ``selected_patches`` : [2D :py:class:`numpy.ndarray`]
            An array of selected patches. The dimensionality of the array:
            (``n_patches`` x ``number_of_features``).
        """

        all_patches = np.vstack(patches)

        idx = [random.randint( 0, len(all_patches) - 1 ) for _ in range(n_patches)]

        selected_patches = all_patches[idx, :]

        return selected_patches


    #==========================================================================
    def select_random_patches(self, frontal_patches, horizontal_patches, vertical_patches, n_patches):
        """
        Select random patches given lists of frontal, central-horizontal and
        central-vertical patches, as returned by ``extract_patches_from_blocks``
        method of this class.

        **Parameters:**

        ``frontal_patches`` : [2D :py:class:`numpy.ndarray`]
            Each element in the list contains an array of vectorized frontal
            patches for the particular stack of facial images.
            The size of each array is:
            ( (``norm_face_size`` - ``block_size``)^2 x ``block_size``^2 ).
            The maximum length of the list is:
            (``num_of_frames_in_video`` - ``block_length``)

        ``horizontal_patches`` : [2D :py:class:`numpy.ndarray`]
            Each element in the list contains an array of vectorized horizontal
            patches for the particular stack of facial images.
            The size of each array is:
            ( (``norm_face_size`` - ``block_size``)^2 x ``block_length``*``block_size`` ).
            The maximum length of the list is:
            (``num_of_frames_in_video`` - ``block_length``)

        ``vertical_patches`` : [2D :py:class:`numpy.ndarray`]
            Each element in the list contains an array of vectorized vertical
            patches for the particular stack of facial images.
            The size of each array is:
            ( (``norm_face_size`` - ``block_size``)^2 x ``block_length``*``block_size`` ).
            The maximum length of the list is:
            (``num_of_frames_in_video`` - ``block_length``)

        ``n_patches`` : :py:class:`int`
            Number of randomly selected patches.
        """

        selected_frontal_patches = self.__select_random_patches_single_list(frontal_patches, n_patches)

        selected_horizontal_patches = self.__select_random_patches_single_list(horizontal_patches, n_patches)

        selected_vertical_patches = self.__select_random_patches_single_list(vertical_patches, n_patches)

        return selected_frontal_patches, selected_horizontal_patches, selected_vertical_patches


    #==========================================================================
    def __call__(self, frames, annotations):
        """
        Do something....

        **Parameters:**

        ``frames`` : FrameContainer
            Video data stored in the FrameContainer, see
            ``bob.bio.video.utils.FrameContainer`` for further details.

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure:
            ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``,
            where
            ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding
            box in frame N.

        **Returns:**

        ``preprocessed_video`` : FrameContainer
            ????????????????
        """

        # Convert frame container to 3D array:
        video = self.convert_frame_cont_to_grayscale_array(frames)

        # get all blocks from all possible facial stacks:
        all_blocks = self.get_all_blocks_from_color_channel(video, annotations,
                                                            self.block_size, self.block_length,
                                                            self.min_face_size, self.norm_face_size)

        frontal_patches, horizontal_patches, vertical_patches = self.extract_patches_from_blocks(all_blocks)

        return frontal_patches, horizontal_patches, vertical_patches


    #==========================================================================
    def write_data( self, frames, file_name ):
        """
        Writes the given data (that has been generated using the __call__
        function of this class) to file. This method overwrites the write_data()
        method of the Preprocessor class.

        **Parameters:**

        ``frames`` :
            data returned by the __call__ method of the class.

        ``file_name`` : :py:class:`str`
            name of the file.
        """

        self.video_preprocessor.write_data(frames, file_name)


    #==========================================================================
    def read_data( self, file_name ):
        """
        Reads the preprocessed data from file.
        This method overwrites the read_data() method of the Preprocessor class.

        **Parameters:**

        ``file_name`` : :py:class:`str`
            name of the file.

        **Returns:**

        ``frames`` : :py:class:`bob.bio.video.FrameContainer`
            Frames stored in the frame container.
        """

        frames = self.video_preprocessor.read_data(file_name)

        return frames


