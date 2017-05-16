#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:14:23 2017

@author: Olegs Nikisins
"""
#==============================================================================
# Import what is needed here:

from bob.bio.base.preprocessor import Preprocessor

from bob.bio.face.preprocessor import FaceCrop

import bob.bio.video

#==============================================================================
# Main body:

class VideoFaceCrop(Preprocessor, object):
    """
    This class is designed to crop faces in each frame of the input video given
    annotations defining the position of the face.

    **Parameters:**

    ``cropped_image_size`` : (int, int)
        The size of the resulting cropped images.

    ``cropped_positions`` : :py:class:`dict`
        The coordinates in the cropped image, where the annotated points should be put to.
        This parameter is a dictionary with usually two elements, e.g., ``{'reye':(RIGHT_EYE_Y, RIGHT_EYE_X) , 'leye':(LEFT_EYE_Y, LEFT_EYE_X)}``.
        However, also other parameters, such as ``{'topleft' : ..., 'bottomright' : ...}`` are supported, as long as the ``annotations`` in the `__call__` function are present.

    ``fixed_positions`` : :py:class:`dict`
        Or None.
        If specified, ignore the annotations from the database and use these fixed positions throughout.

    ``mask_sigma`` : :py:class:`float`
        Or None
        Fill the area outside of image boundaries with random pixels from the border, by adding noise to the pixel values.
        To disable extrapolation, set this value to ``None``.
        To disable adding random noise, set it to a negative value or 0.

    ``mask_neighbors`` : :py:class:`int`
        The number of neighbors used during mask extrapolation.
        See :py:func:`bob.ip.base.extrapolate_mask` for details.

    ``mask_seed`` : :py:class:`int`
        Or None.
        The random seed to apply for mask extrapolation.

        .. warning::
          When run in parallel, the same random seed will be applied to all parallel processes.
          Hence, results of parallel execution will differ from the results in serial execution.

    ``kwargs``
        Remaining keyword parameters passed to the :py:class:`Base` constructor, such as ``color_channel`` or ``dtype``.

    """

    #==========================================================================
    def __init__(self,
                 cropped_image_size,
                 cropped_positions,
                 fixed_positions = None,
                 mask_sigma = None,
                 mask_neighbors = 5,
                 mask_seed = None,
                 **kwargs):

        super(VideoFaceCrop, self).__init__(cropped_image_size = cropped_image_size,
                                             cropped_positions = cropped_positions,
                                             fixed_positions = fixed_positions,
                                             mask_sigma = mask_sigma,
                                             mask_neighbors = mask_neighbors,
                                             mask_seed = mask_seed,
                                             **kwargs)

        self.cropped_image_size = cropped_image_size
        self.cropped_positions = cropped_positions
        self.fixed_positions = fixed_positions
        self.mask_sigma = mask_sigma
        self.mask_neighbors = mask_neighbors
        self.mask_seed = mask_seed

        # Save also the data stored in the kwargs:
        for (k, v) in kwargs.items():
            setattr(self, k, v)

        preprocessor = FaceCrop(cropped_image_size = cropped_image_size,
                                cropped_positions = cropped_positions,
                                fixed_positions = fixed_positions,
                                mask_sigma = mask_sigma,
                                mask_neighbors = mask_neighbors,
                                mask_seed = mask_seed,
                                **kwargs)

        self.video_preprocessor = bob.bio.video.preprocessor.Wrapper(preprocessor)


    #==========================================================================
    def __call__(self, frames, annotations):
        """
        Crop the face in the input video frames given annotations for each frame.

        **Parameters:**

        ``frames`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box in frame N.

        **Returns:**

        ``preprocessed_video`` : FrameContainer
            Cropped faces stored in the FrameContainer.
        """

        preprocessed_video = self.video_preprocessor(frames = frames, annotations = annotations)

        return preprocessed_video


    #==========================================================================
    def write_data( self, frames, file_name ):
        """
        Writes the given data (that has been generated using the __call__ function of this class) to file.
        This method overwrites the write_data() method of the Preprocessor class.

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


