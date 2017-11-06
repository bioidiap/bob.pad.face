#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
This file contains face detection utils.
"""
#==============================================================================
# Import here:

import importlib


#==============================================================================
def detect_face_in_image(image, method = "dlib"):
    """
    This function detects a face in the input image.

    **Parameters:**

    ``image`` : 3D :py:class:`numpy.ndarray`
        A color image to detect the face in.

    ``method`` : :py:class:`str`
        A package to be used for face detection. Options supported by this
        package: "dlib" (dlib is a dependency of this package). If  bob.ip.mtcnn
        is installed in your system you can use it as-well (bob.ip.mtcnn is NOT
        a dependency of this package).

    **Returns:**

    ``annotations`` : :py:class:`dict`
        A dictionary containing annotations of the face bounding box.
        Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col)}``.
        If no annotations found an empty dictionary is returned.
    """

    try:
        face_detection_module = importlib.import_module("bob.ip." + method)
    except ImportError:
        raise ImportError("No module named bob.ip." + method)

    if not hasattr(face_detection_module, 'FaceDetector'):
        raise AttributeError("bob.ip." + method + " module has no attribute FaceDetector")

    data = face_detection_module.FaceDetector().detect_single_face(image)

    annotations = {}

    if ( data is not None ) and ( not all([x is None for x in data]) ):

        bounding_box = data[0]

        annotations['topleft'] = bounding_box.topleft

        annotations['bottomright'] = bounding_box.bottomright

    return annotations


#==============================================================================
def detect_faces_in_video(frame_container, method = "dlib"):
    """
    This function detects a face in each farme of the input video.

    **Parameters:**

    ``frame_container`` : FrameContainer
        FrameContainer containing the frames data.

    ``method`` : :py:class:`str`
        A package to be used for face detection. Options supported by this
        package: "dlib" (dlib is a dependency of this package). If  bob.ip.mtcnn
        is installed in your system you can use it as-well (bob.ip.mtcnn is NOT
        a dependency of this package).

    **Returns:**

    ``annotations`` : :py:class:`dict`
        A dictionary containing the annotations for each frame in the video.
        Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
        Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
        is the dictionary defining the coordinates of the face bounding box in frame N.
        If no annotations found an empty dictionary is returned.
    """

    annotations = {}

    for idx, frame in enumerate(frame_container):

        image = frame[1]

        frame_annotations = detect_face_in_image(image, method)

        if frame_annotations:

            annotations[str(idx)] = frame_annotations

    return annotations









