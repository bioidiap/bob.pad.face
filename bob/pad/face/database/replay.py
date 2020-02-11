#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.pad.base.database import PadDatabase  # Used in ReplayMobilePadFile class
from bob.pad.face.database import VideoPadFile  # Used in ReplayPadFile class
from bob.extension import rc
from bob.ip.facedetect import expected_eye_positions, BoundingBox
from bob.db.base.annotations import read_annotation_file

REPLAY_ATTACK_FRAME_SHAPE = (3, 240, 320)


class ReplayPadFile(VideoPadFile):
    """
    A high level implementation of the File class for the REPLAY-ATTACK
    database.
    """

    def __init__(self, f):
        """
        Parameters
        ----------
        f : object
            An instance of the File class defined in the low level db interface
            of the Replay database, in the bob.db.replay.models.py file.
        """

        self.f = f
        # this f is actually an instance of the File class that is defined in
        # bob.db.replay.models and the PadFile class here needs client_id,
        # path, attack_type, file_id for initialization. We have to convert
        # information here and provide them to PadFile. attack_type is a little
        # tricky to get here. Based on the documentation of PadFile: In cased
        # of a spoofed data, this parameter should indicate what kind of
        # spoofed attack it is. The default None value is interpreted that the
        # PadFile is a genuine or real sample.
        if f.is_real():
            attack_type = None
        else:
            attack_type = "attack"
        # attack_type is a string and I decided to make it like this for this
        # particular database. You can do whatever you want for your own
        # database.

        super(ReplayPadFile, self).__init__(
            client_id=f.client_id, path=f.path, attack_type=attack_type, file_id=f.id
        )

    @property
    def frame_shape(self):
        """Returns the size of each frame in this database.

        Returns
        -------
        (int, int, int)
            The (#Channels, Height, Width) which is (3, 240, 320).
        """
        return REPLAY_ATTACK_FRAME_SHAPE

    @property
    def annotations(self):
        """
        Return annotations as a dictionary of dictionaries.

        If the file object has an attribute of annotation_directory, it will read
        annotations from there instead of loading annotations that are shipped with the
        database.

        Returns
        -------
        annotations : :py:class:`dict`
            A dictionary containing the annotations for each frame in the
            video. Dictionary structure:
            ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.Where
            ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box
            in frame N.
        """
        if (
            hasattr(self, "annotation_directory")
            and self.annotation_directory is not None
        ):
            path = self.make_path(self.annotation_directory, extension=".json")
            return read_annotation_file(path, annotation_type="json")

        # numpy array containing the face bounding box data for each video
        # frame, returned data format described in the f.bbx() method of the
        # low level interface
        annots = self.f.bbx(directory=self.original_directory)

        annotations = {}  # dictionary to return

        for fn, frame_annots in enumerate(annots):

            topleft = (frame_annots[2], frame_annots[1])
            bottomright = (
                frame_annots[2] + frame_annots[4],
                frame_annots[1] + frame_annots[3],
            )
            annotations[str(fn)] = {"topleft": topleft, "bottomright": bottomright}

            size = (bottomright[0] - topleft[0], bottomright[1] - topleft[1])
            bounding_box = BoundingBox(topleft, size)
            annotations[str(fn)].update(expected_eye_positions(bounding_box))

        return annotations


class ReplayPadDatabase(PadDatabase):
    """
    A high level implementation of the Database class for the REPLAY-ATTACK
    database.
    """

    def __init__(
        self,
        # grandtest is the default protocol for this database
        protocol="grandtest",
        original_directory=rc["bob.db.replay.directory"],
        original_extension=".mov",
        annotation_directory=None,
        **kwargs
    ):
        """
        Parameters
        ----------

        protocol : str or None
            The name of the protocol that defines the default experimental
            setup for this database.

        original_directory : str
            The directory where the original data of the database are stored.

        original_extension : str
            The file name extension of the original data.

        kwargs
            The arguments of the :py:class:`bob.bio.base.database.BioDatabase`
            base class constructor.
        """

        from bob.db.replay import Database as LowLevelDatabase

        self.db = LowLevelDatabase()

        # Since the high level API expects different group names than what the
        # low level API offers, you need to convert them when necessary
        self.low_level_group_names = (
            "train",
            "devel",
            "test",
        )  # group names in the low-level database interface
        self.high_level_group_names = (
            "train",
            "dev",
            "eval",
        )  # names are expected to be like that in objects() function

        # Always use super to call parent class methods.
        super(ReplayPadDatabase, self).__init__(
            name="replay",
            protocol=protocol,
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            **kwargs
        )

    @property
    def original_directory(self):
        return self.db.original_directory

    @original_directory.setter
    def original_directory(self, value):
        self.db.original_directory = value

    def objects(
        self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs
    ):
        """
        This function returns lists of ReplayPadFile objects, which fulfill the
        given restrictions.

        Parameters
        ----------
        groups : :obj:`str` or [:obj:`str`]
            The groups of which the clients should be returned.
            Usually, groups are one or more elements of
            ('train', 'dev', 'eval')

        protocol : str
            The protocol for which the clients should be retrieved.
            The protocol is dependent on your database.
            If you do not have protocols defined, just ignore this field.

        purposes : :obj:`str` or [:obj:`str`]
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.

        model_ids
            This parameter is not supported in PAD databases yet
        **kwargs

        Returns
        -------
        files : [ReplayPadFile]
            A list of ReplayPadFile objects.
        """

        # Convert group names to low-level group names here.
        groups = self.convert_names_to_lowlevel(
            groups, self.low_level_group_names, self.high_level_group_names
        )
        # Since this database was designed for PAD experiments, nothing special
        # needs to be done here.
        files = self.db.objects(
            protocol=protocol, groups=groups, cls=purposes, **kwargs
        )
        files = [ReplayPadFile(f) for f in files]
        # set the attributes
        for f in files:
            f.original_directory = self.original_directory
            f.original_extension = self.original_extension
            f.annotation_directory = self.annotation_directory
        return files

    def annotations(self, f):
        """
        Return annotations for a given file object ``f``, which is an instance
        of ``ReplayPadFile`` defined in the HLDI of the Replay-Attack DB. The
        ``load()`` method of ``ReplayPadFile`` class (see above) returns a
        video, therefore this method returns bounding-box annotations for each
        video frame. The annotations are returned as a dictionary of
        dictionaries.

        Parameters
        ----------
        f : :any:`ReplayPadFile`
            An instance of :any:`ReplayPadFile`.

        Returns
        -------
        annotations : :py:class:`dict`
            A dictionary containing the annotations for each frame in the
            video. Dictionary structure:
            ``annotations = {'1': frame1_dict, '2': frame1_dict, ...}``.Where
            ``frameN_dict = {'topleft': (row, col), 'bottomright': (row, col)}``
            is the dictionary defining the coordinates of the face bounding box
            in frame N.
        """
        return f.annotations

    def frames(self, padfile):
        """Yields the frames of the padfile one by one.

        Parameters
        ----------
        padfile : :any:`ReplayPadFile`
            The high-level replay pad file

        Yields
        ------
        :any:`numpy.array`
            A frame of the video. The size is (3, 240, 320).
        """
        return padfile.frames

    def number_of_frames(self, padfile):
        """Returns the number of frames in a video file.

        Parameters
        ----------
        padfile : :any:`ReplayPadFile`
            The high-level pad file

        Returns
        -------
        int
            The number of frames.
        """
        return padfile.number_of_frames

    @property
    def frame_shape(self):
        """Returns the size of each frame in this database.

        Returns
        -------
        (int, int, int)
            The (#Channels, Height, Width) which is (3, 240, 320).
        """
        return REPLAY_ATTACK_FRAME_SHAPE
