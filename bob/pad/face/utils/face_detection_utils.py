#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
This file contains face detection utils.
"""
#==============================================================================
# Import here:

import bob.ip.dlib # for face detection functionality


#==============================================================================
def detect_face_in_image(image):
    """
    This function detects a face in the input image.

    **Parameters:**

    ``image`` : 3D :py:class:`numpy.ndarray`
        A color image to detect the face in.

    **Returns:**

    ``annotations`` : :py:class:`dict`
        A dictionary containing annotations of the face bounding box.
        Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col)}``.
    """

    bounding_box, _ = bob.ip.dlib.FaceDetector().detect_single_face(image)

    annotations = {}

    if bounding_box is not None:

        annotations['topleft'] = bounding_box.topleft

        annotations['bottomright'] = bounding_box.bottomright

    else:

        annotations['topleft'] = (0, 0)

        annotations['bottomright'] = (0, 0)

    return annotations


#==============================================================================
def detect_faces_in_video(frame_container):
    """
    This function detects a face in each farme of the input video.

    **Parameters:**

    ``frame_container`` : FrameContainer
        FrameContainer containing the frames data.

    **Returns:**

    ``annotations`` : :py:class:`dict`
        A dictionary containing the annotations for each frame in the video.
        Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
        Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
        is the dictionary defining the coordinates of the face bounding box in frame N.
    """

    annotations = {}

    for idx, frame in enumerate(frame_container):

        image = frame[1]

        frame_annotations = detect_face_in_image(image)

        annotations[str(idx)] = frame_annotations

    return annotations









