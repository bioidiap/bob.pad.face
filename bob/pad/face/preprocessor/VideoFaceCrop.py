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

import numpy as np

from bob.pad.face.preprocessor.ImageFaceCrop import ImageFaceCrop

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

    ``check_face_size_flag`` : :py:class:`bool`
        If True, only return the frames containing faces of the size above the
        specified threshold ``min_face_size``. Default: False.

    ``min_face_size`` : :py:class:`int`
        The minimal size of the face in pixels. Only valid when ``check_face_size_flag``
        is set to True. Default: 50.

    ``use_local_cropper_flag`` : :py:class:`bool`
        If True, use the local ImageFaceCrop class to crop faces in the frames.
        Otherwise, the FaceCrop preprocessor from bob.bio.face is used.
        Default: False.

    ``rgb_output_flag`` : :py:class:`bool`
        Return RGB cropped face if ``True``, otherwise a gray-scale image is
        returned. This flag is only valid when ``use_local_cropper_flag = True``.
        Default: ``False``.

    ``kwargs``
        Remaining keyword parameters passed to the Base constructor, such as ``color_channel`` or ``dtype``.
    """

    #==========================================================================
    def __init__(self,
                 cropped_image_size,
                 cropped_positions,
                 fixed_positions = None,
                 mask_sigma = None,
                 mask_neighbors = 5,
                 mask_seed = None,
                 check_face_size_flag = False,
                 min_face_size = 50,
                 use_local_cropper_flag = False,
                 rgb_output_flag = False,
                 **kwargs):

        super(VideoFaceCrop, self).__init__(cropped_image_size = cropped_image_size,
                                            cropped_positions = cropped_positions,
                                            fixed_positions = fixed_positions,
                                            mask_sigma = mask_sigma,
                                            mask_neighbors = mask_neighbors,
                                            mask_seed = mask_seed,
                                            check_face_size_flag = check_face_size_flag,
                                            min_face_size = min_face_size,
                                            use_local_cropper_flag = use_local_cropper_flag,
                                            rgb_output_flag = rgb_output_flag,
                                            **kwargs)

        self.cropped_image_size = cropped_image_size
        self.cropped_positions = cropped_positions
        self.fixed_positions = fixed_positions
        self.mask_sigma = mask_sigma
        self.mask_neighbors = mask_neighbors
        self.mask_seed = mask_seed
        self.check_face_size_flag = check_face_size_flag
        self.min_face_size = min_face_size
        self.use_local_cropper_flag = use_local_cropper_flag
        self.rgb_output_flag = rgb_output_flag

        # Save also the data stored in the kwargs:
        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.use_local_cropper_flag:

            preprocessor = ImageFaceCrop(face_size = self.cropped_image_size[0],
                                         rgb_output_flag = self.rgb_output_flag)

        else:

            preprocessor = FaceCrop(cropped_image_size = self.cropped_image_size,
                                    cropped_positions = self.cropped_positions,
                                    fixed_positions = self.fixed_positions,
                                    mask_sigma = self.mask_sigma,
                                    mask_neighbors = self.mask_neighbors,
                                    mask_seed = self.mask_seed,
                                    **kwargs)

        self.video_preprocessor = bob.bio.video.preprocessor.Wrapper(preprocessor)


    #==========================================================================
    def check_face_size(self, frame_container, annotations, min_face_size):
        """
        Return the FrameContainer containing the frames with faces of the
        size overcoming the specified threshold.

        **Parameters:**

        ``frame_container`` : FrameContainer
            Video data stored in the FrameContainer, see ``bob.bio.video.utils.FrameContainer``
            for further details.

        ``annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the video.
            Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box in frame N.

        ``min_face_size`` : :py:class:`int`
            The minimal size of the face in pixels.

        **Returns:**

        ``cleaned_frame_container`` : FrameContainer
            FrameContainer containing the frames with faces of the size
            overcoming the specified threshold.
        """

        cleaned_frame_container = bob.bio.video.FrameContainer() # initialize the FrameContainer

        selected_frame_idx = 0

        for idx in range(0, np.min( [len(annotations), len(frame_container)] )): # idx - frame index

            frame_annotations = annotations[str(idx)] # annotations for particular frame

            # size of current face
            face_size = np.min(np.array(frame_annotations['bottomright']) - np.array(frame_annotations['topleft']))

            if face_size >= min_face_size: # check if face size is above the threshold

                selected_frame = frame_container[idx][1] # get current frame

                cleaned_frame_container.add(selected_frame_idx, selected_frame) # add current frame to FrameContainer

                selected_frame_idx = selected_frame_idx + 1

        return cleaned_frame_container


    #==========================================================================
    def select_annotated_frames(self, frames, annotations):
        """
        Select only annotated frames in the input FrameContainer ``frames``.

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

        ``cleaned_frame_container`` : FrameContainer
            FrameContainer containing the annotated frames only.

        ``cleaned_annotations`` : :py:class:`dict`
            A dictionary containing the annotations for each frame in the output video.
            Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
            Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box in frame N.
        """

        annotated_frames = np.sort( [np.int(item) for item in annotations.keys()] ) # annotated frame numbers

        available_frames = range(0,len(frames)) # frame numbers in the input video

        valid_frames = list(set(annotated_frames).intersection(available_frames)) # valid and annotated frames

        cleaned_frame_container = bob.bio.video.FrameContainer() # initialize the FrameContainer

        cleaned_annotations = {}

        for idx, valid_frame_num in enumerate(valid_frames):
            ## valid_frame_num - is the number of the original frame having annotations

            cleaned_annotations[str(idx)] = annotations[str(valid_frame_num)] # correct the frame numbers

            selected_frame = frames[valid_frame_num][1] # get current frame

            cleaned_frame_container.add(idx, selected_frame) # add current frame to FrameContainer

        return cleaned_frame_container, cleaned_annotations


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

        if len(frames) != len(annotations): # if some annotations are missing

            ## Select only annotated frames:
            frames, annotations = self.select_annotated_frames(frames, annotations)

        preprocessed_video = self.video_preprocessor(frames = frames, annotations = annotations)

        if self.check_face_size_flag:

            preprocessed_video = self.check_face_size(preprocessed_video, annotations, self.min_face_size)

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


