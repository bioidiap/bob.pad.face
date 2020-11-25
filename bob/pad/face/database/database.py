from bob.pad.base.database import PadFile
import bob.bio.video
import bob.io.video
from bob.db.base.annotations import read_annotation_file


class VideoPadFile(PadFile):
    """A simple base class that defines basic properties of File object for the
    use in face PAD experiments.
    """

    def __init__(
        self,
        attack_type,
        client_id,
        path,
        file_id=None,
        original_directory=None,
        original_extension=".avi",
        annotation_directory=None,
        annotation_extension=None,
        annotation_type=None,
        selection_style=None,
        max_number_of_frames=None,
        step_size=None,
    ):
        super().__init__(
            attack_type=attack_type,
            client_id=client_id,
            path=path,
            file_id=file_id,
            original_directory=original_directory,
            original_extension=original_extension,
            annotation_directory=annotation_directory,
            annotation_extension=annotation_extension,
            annotation_type=annotation_type,
        )
        self.selection_style = selection_style or "all"
        self.max_number_of_frames = max_number_of_frames
        self.step_size = step_size

    def load(
        self,
    ):
        """Loads the video file and returns in a `bob.bio.video.FrameContainer`.

        Returns
        -------
        :any:`bob.bio.video.VideoAsArray`
            The loaded frames inside a frame container.
        """
        path = self.make_path(self.original_directory, self.original_extension)
        video = bob.bio.video.VideoAsArray(
            path,
            selection_style=self.selection_style,
            max_number_of_frames=self.max_number_of_frames,
            step_size=self.step_size,
        )
        return video

    @property
    def frames(self):
        """Returns an iterator of frames in the video.
        If your database video files need to be loaded in a special way, you need to
        override this property.

        Returns
        -------
        collection.Iterator
            An iterator returning frames of the video.
        """
        path = self.make_path(
            directory=self.original_directory, extension=self.original_extension
        )
        return iter(bob.io.video.reader(path))

    @property
    def number_of_frames(self):
        path = self.make_path(
            directory=self.original_directory, extension=self.original_extension
        )
        return bob.io.video.reader(path).number_of_frames

    @property
    def frame_shape(self):
        """Returns the size of each frame in this database.
        This implementation assumes all frames have the same shape.
        It's best to override this method in your database implementation and return
        a constant.

        Returns
        -------
        (int, int, int)
            The (Channels, Height, Width) sizes.
        """
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
        if self.annotation_directory is None:
            return None

        annotation_file = self.make_path(
            directory=self.annotation_directory, extension=self.annotation_extension
        )
        return read_annotation_file(annotation_file, self.annotation_type)
