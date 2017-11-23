#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
"""
This file contains face detection utils.
"""
#==============================================================================
# Import here:

import importlib
import numpy as np


#==============================================================================
def get_eye_pos(lm):
    """
    This function returns the locations of left and right eyes

    **Parameters:**

    ``lm`` : :py:class:`array`
        A numpy array containing the coordinates of facial landmarks, (68X2)

    **Returns:**

    ``right_eye``
        A tuple containing the location of right eye,

    ``left_eye``
        A tuple containing the location of left eye

    """

    # Mean position of eye corners as eye centers , casted to int()

    left_eye_t = (lm[36, :] + lm[39, :]) / 2.0
    right_eye_t = (lm[42, :] + lm[45, :]) / 2.0

    right_eye = (int(left_eye_t[1]), int(left_eye_t[0]))
    left_eye = (int(right_eye_t[1]), int(right_eye_t[0]))

    return right_eye, left_eye


#==============================================================================
def detect_face_in_image(image, method="dlib"):
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
        raise AttributeError(
            "bob.ip." + method + " module has no attribute FaceDetector")

    data = face_detection_module.FaceDetector().detect_single_face(image)

    annotations = {}

    if (data is not None) and (not all([x is None for x in data])):

        bounding_box = data[0]

        annotations['topleft'] = bounding_box.topleft

        annotations['bottomright'] = bounding_box.bottomright

    return annotations


#==============================================================================
def detect_faces_in_video(frame_container, method="dlib"):
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


#==============================================================================
def detect_face_landmarks_in_image(image, method="dlib"):
    """
    This function detects a face in the input image. Two oprions for face detector , but landmark detector is always the same

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
        A dictionary containing annotations of the face bounding box, eye locations and facial landmarks.
        Dictionary must be as follows ``{'topleft': (row, col), 'bottomright': (row, col), 'left_eye': (row, col), 'right_eye': (row, col), 'landmarks': [(col1,row1), (col2,row2), ...]}``.
        If no annotations found an empty dictionary is returned.
        Where (rowK,colK) is the location of Kth facial landmark (K=0,...,67).
    """

    ### Face detector

    try:
        face_detection_module = importlib.import_module("bob.ip." + method)

    except ImportError:

        print("No module named bob.ip." + method +
              " trying to use default method!")

        try:
            face_detection_module = importlib.import_module("bob.ip.dlib")
            method = "dlib"
        except ImportError:
            raise ImportError("No module named bob.ip.dlib")

    if not hasattr(face_detection_module, 'FaceDetector'):
        raise AttributeError(
            "bob.ip." + method + " module has no attribute FaceDetector!")

    #### Landmark detector

    try:
        landmark_detection_module = importlib.import_module(
            "bob.ip.facelandmarks")
    except ImportError:
        raise ImportError("No module named bob.ip.facelandmarks!!")

    if not hasattr(landmark_detection_module,
                   'detect_landmarks_on_boundingbox'):
        raise AttributeError(
            "bob.ip.facelandmarksmodule has no attribute detect_landmarks_on_boundingbox!"
        )

    face_detector = face_detection_module.FaceDetector()

    data = face_detector.detect_single_face(image)

    annotations = {}

    if (data is not None) and (not all([x is None for x in data])):

        bounding_box = data[0]

        bounding_box_scaled = bounding_box.scale(0.95, True)  # is ok for dlib

        lm = landmark_detection_module.detect_landmarks_on_boundingbox(
            image, bounding_box_scaled)

        if lm is not None:

            lm = np.array(lm)

            lm = np.vstack((lm[:, 1], lm[:, 0])).T

            #print("LM",lm)

            right_eye, left_eye = get_eye_pos(lm)

            points = []

            for i in range(lm.shape[0]):

                points.append((int(lm[i, 0]), int(lm[i, 1])))

            annotations['topleft'] = bounding_box.topleft

            annotations['bottomright'] = bounding_box.bottomright

            annotations['landmarks'] = points

            annotations['left_eye'] = left_eye

            annotations['right_eye'] = right_eye

    return annotations


#==============================================================================
def detect_face_landmarks_in_video(frame_container, method="dlib"):
    """
    This function detects a face and face landmarks  in each farme of the input video.

    **Parameters:**

    ``frame_container`` : FrameContainer
        FrameContainer containing the frames data.

    ``method`` : :py:class:`str`
        A package to be used for face detection. Options supported by this
        package: "dlib" (dlib is a dependency of this package) and "mtcnn".
        If its found it will be used else defaults to "dlib"

    **Returns:**

    ``annotations`` : :py:class:`dict`
        A dictionary containing the annotations for each frame in the video.
        Dictionary structure: ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.
        Where ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col), 'left_eye': (row, col), 'right_eye': (row, col), 'landmarks': [(col1,row1), (col2,row2), ...]}``
        is the dictionary defining the coordinates of the face bounding box in frame N.
        Where (colK,rowK) is the location of Kth facial landmark (K=0,...,67).
        If no annotations found an empty dictionary is returned.
    """

    annotations = {}

    for idx, frame in enumerate(frame_container):

        image = frame[1]

        frame_annotations = detect_face_landmarks_in_image(image, method)

        if frame_annotations:

            annotations[str(idx)] = frame_annotations

    return annotations
