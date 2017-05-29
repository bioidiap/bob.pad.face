#!/usr/bin/env python

from bob.pad.face.preprocessor import VideoFaceCrop


#=======================================================================================
# Define instances here:

cropped_image_size = (64, 64) # The size of the resulting face
cropped_positions = {'topleft' : (0,0) , 'bottomright' : cropped_image_size}
fixed_positions = None
mask_sigma = None             # The sigma for random values areas outside image
mask_neighbors = 5            # The number of neighbors to consider while extrapolating
mask_seed = None              # The seed for generating random values during extrapolation
color_channel = 'gray'        # Convert image to gray-scale format

video_face_crop_preproc_64_64 = VideoFaceCrop(cropped_image_size = cropped_image_size,
                                                cropped_positions = cropped_positions,
                                                fixed_positions = fixed_positions,
                                                mask_sigma = mask_sigma,
                                                mask_neighbors = mask_neighbors,
                                                mask_seed = None,
                                                color_channel = color_channel)
