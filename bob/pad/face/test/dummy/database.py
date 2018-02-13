from bob.bio.base.test.utils import atnt_database_directory
from bob.bio.video.utils import FrameContainer
import bob.io.base
import os
from bob.pad.face.database import VideoPadFile
from bob.pad.base.database import PadDatabase
from bob.db.base.utils import check_parameters_for_validity, convert_names_to_lowlevel


class DummyPadFile(VideoPadFile):
    def load(self, directory=None, extension='.pgm', frame_selector=None):
        file_name = self.make_path(directory, extension)
        fc = FrameContainer()
        fc.add(os.path.basename(file_name), bob.io.base.load(file_name))
        return fc


class DummyDatabase(PadDatabase):

    def __init__(self):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__(
            name='test',
            original_directory=atnt_database_directory(),
            original_extension='.pgm',
            check_original_files_for_existence=True,
            training_depends_on_protocol=False,
            models_depend_on_protocol=False
        )
        import bob.db.atnt
        self._db = bob.db.atnt.Database()
        self.low_level_names = ('world', 'dev')
        self.high_level_names = ('train', 'dev')

    def _make_bio(self, files):
        return [DummyPadFile(client_id=f.client_id, path=f.path, file_id=f.id,
                             attack_type=None)
                for f in files]

    def objects(self, groups=None, protocol=None, purposes=None,
                model_ids=None, **kwargs):
        groups = check_parameters_for_validity(
            groups, 'groups', self.high_level_names, default_parameters=None)
        groups = convert_names_to_lowlevel(
            groups, self.low_level_names, self.high_level_names)
        purposes = list(check_parameters_for_validity(
            purposes, 'purposes', ('real', 'attack'),
            default_parameters=('real', 'attack')))
        if 'real' in purposes:
            purposes.remove('real')
            purposes.append('enroll')
        if 'attack' in purposes:
            purposes.remove('attack')
            purposes.append('probe')
        return self._make_bio(self._db.objects(model_ids, groups, purposes,
                                               protocol, **kwargs))

    def annotations(self, file):
        return None

    def frames(self, padfile):
        fc = padfile.load(self.original_directory)
        for _, frame, _ in fc:
            yield frame

    def number_of_frames(self, padfile):
        fc = padfile.load(self.original_directory)
        return len(fc)

    @property
    def frame_shape(self):
        return (112, 92)


database = DummyDatabase()
