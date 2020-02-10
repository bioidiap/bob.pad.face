from bob.pad.base.database import PadFile
import bob.bio.video
import bob.io.video
from bob.db.base.annotations import read_annotation_file


class VideoPadFile(PadFile):
    """A simple base class that defines basic properties of File object for the
    use in face PAD experiments.
    """

    def __init__(self, attack_type, client_id, path, file_id=None):
        super(VideoPadFile, self).__init__(
            attack_type=attack_type, client_id=client_id, path=path, file_id=file_id
        )

    def load(
        self,
        directory=None,
        extension=".avi",
        frame_selector=bob.bio.video.FrameSelector(selection_style="all"),
    ):
        """Loads the video file and returns in a :any:`bob.bio.video.FrameContainer`.

        Parameters
        ----------
        directory : :obj:`str`, optional
            The directory to load the data from.
        extension : :obj:`str`, optional
            The extension of the file to load.
        frame_selector : :any:`bob.bio.video.FrameSelector`, optional
            Which frames to select.

        Returns
        -------
        :any:`bob.bio.video.FrameContainer`
            The loaded frames inside a frame container.
        """
        return frame_selector(self.make_path(directory, extension))

    def check_original_directory_and_extension(self):
        if not hasattr(self, "original_directory"):
            raise RuntimeError(
                "Please set the original_directory attribute of files in your "
                "database's object method."
            )
        if not hasattr(self, "original_extension"):
            raise RuntimeError(
                "Please set the original_extension attribute of files in your "
                "database's object method."
            )

    @property
    def frames(self):
        """Returns an iterator of frames in the video.
        If your database video files need to be loaded in a special way, you need to
        override this property.

        Returns
        -------
        collection.Iterator
            An iterator returning frames of the video.

        Raises
        ------
        RuntimeError
            In your database implementation, the original_directory and
            original_extension attributes of the files need to be set when database's
            object method is called.
        """
        self.check_original_directory_and_extension()
        path = self.make_path(
            directory=self.original_directory, extension=self.original_extension
        )
        return iter(bob.io.video.reader(path))

    @property
    def number_of_frames(self):
        self.check_original_directory_and_extension()
        path = self.make_path(
            directory=self.original_directory, extension=self.original_extension
        )
        return bob.io.video.reader(path).number_of_frames

    @property
    def frame_shape(self):
        """Returns the size of each frame in this database.
        This implementation assumes all videos and frames have the same shape.
        It's best to override this method in your database implementation and return
        a constant.

        Returns
        -------
        (int, int, int)
            The (Channels, Height, Width) sizes.
        """
        self.check_original_directory_and_extension()
        path = self.make_path(
            directory=self.original_directory, extension=self.original_extension
        )
        frame = next(bob.io.video.reader(path))
        return frame.shape

    @property
    def annotations(self):
        """Reads the annotations
        For this property to work, you need to set ``annotation_directory``,
        ``annotation_extension``, and ``annotation_type`` attributes of the files when
        database's object method is called.

        Returns
        -------
        dict
            The annotations as a dictionary.
        """
        if not (
            hasattr(self, "annotation_directory")
            and hasattr(self, "annotation_extension")
            and hasattr(self, "annotation_type")
        ):
            raise RuntimeError(
                "For this property to work, you need to set ``annotation_directory``, "
                "``annotation_extension``, and ``annotation_type`` attributes of the "
                "files when database's object method is called."
            )

        if self.annotation_directory is None:
            return None

        annotation_file = self.make_path(
            directory=self.annotation_directory, extension=self.annotation_extension
        )
        return read_annotation_file(annotation_file, self.annotation_type)
