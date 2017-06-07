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
check_face_size_flag = False  # Don't check the size of the face
min_face_size = 50
use_local_cropper_flag = False # Use the cropper of bob.bio.face
color_channel = 'gray'        # Convert image to gray-scale format

video_face_crop_preproc_64_64 = VideoFaceCrop(cropped_image_size = cropped_image_size,
                                              cropped_positions = cropped_positions,
                                              fixed_positions = fixed_positions,
                                              mask_sigma = mask_sigma,
                                              mask_neighbors = mask_neighbors,
                                              mask_seed = None,
                                              check_face_size_flag = check_face_size_flag,
                                              min_face_size = min_face_size,
                                              use_local_cropper_flag = use_local_cropper_flag,
                                              color_channel = color_channel)

check_face_size_flag = True  # Check the size of the face
min_face_size = 50

video_face_crop_preproc_64_64_face_50 = VideoFaceCrop(cropped_image_size = cropped_image_size,
                                                      cropped_positions = cropped_positions,
                                                      fixed_positions = fixed_positions,
                                                      mask_sigma = mask_sigma,
                                                      mask_neighbors = mask_neighbors,
                                                      mask_seed = None,
                                                      check_face_size_flag = check_face_size_flag,
                                                      min_face_size = min_face_size,
                                                      use_local_cropper_flag = use_local_cropper_flag,
                                                      color_channel = color_channel)


use_local_cropper_flag = True # Use the local face cropping class (identical to Ivana's paper)

video_face_crop_preproc_64_64_face_50_local_cropper = VideoFaceCrop(cropped_image_size = cropped_image_size,
                                                                    cropped_positions = cropped_positions,
                                                                    fixed_positions = fixed_positions,
                                                                    mask_sigma = mask_sigma,
                                                                    mask_neighbors = mask_neighbors,
                                                                    mask_seed = None,
                                                                    check_face_size_flag = check_face_size_flag,
                                                                    min_face_size = min_face_size,
                                                                    use_local_cropper_flag = use_local_cropper_flag,
                                                                    color_channel = color_channel)

rgb_output_flag = True # Return RGB cropped face using local cropper

video_face_crop_preproc_64_64_face_50_local_cropper_rgb = VideoFaceCrop(cropped_image_size = cropped_image_size,
                                                                    cropped_positions = cropped_positions,
                                                                    fixed_positions = fixed_positions,
                                                                    mask_sigma = mask_sigma,
                                                                    mask_neighbors = mask_neighbors,
                                                                    mask_seed = None,
                                                                    check_face_size_flag = check_face_size_flag,
                                                                    min_face_size = min_face_size,
                                                                    use_local_cropper_flag = use_local_cropper_flag,
                                                                    rgb_output_flag = rgb_output_flag)