"""The `Mask-Attack`_ database for face anti-spoofing
consists of video clips of mask attacks. This database was produced at the
`Idiap Research Institute <http://www.idiap.ch>`_, in Switzerland.

If you use this database in your publication, please cite the following paper on
your references:


    @INPROCEEDINGS{ERDOGMUS_BTAS-2013,
        author = {Erdogmus, Nesli and Marcel, Sébastien},
        keywords = {biometric, Counter-Measures, Spoofing Attacks},
        month = september,
        title = {Spoofing in 2D Face Recognition with 3D Masks and Anti-spoofing with Kinect},
        journal = {Biometrics: Theory, Applications and Systems},
        year = {2013},}

After downloading, you can tell the bob library where the files are located
using::

    $ bob config set bob.db.mask_attack.directory /path/to/database/3dmad/Data/
"""
from bob.pad.face.database import MaskAttackPadDatabase

database = MaskAttackPadDatabase()
