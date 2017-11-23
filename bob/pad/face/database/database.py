from bob.pad.base.database import PadFile
from bob.bio.video.database import VideoBioFile


class VideoPadFile(VideoBioFile, PadFile):
    """A simple base class that defines basic properties of File object for the
    use in PAD experiments
    """

    def __init__(self, attack_type, client_id, path, file_id=None):
        VideoBioFile.__init__(
            self,
            client_id=client_id,
            path=path,
            file_id=file_id,
        )
        PadFile.__init__(
            self,
            attack_type=attack_type,
            client_id=client_id,
            path=path,
            file_id=file_id,
        )
