"""The Swan_ Database.

To configure the location of the database on your computer, run::

    bob config set bob.db.swan.directory /path/to/database/swan


The Idiap part of the dataset comprises 150 subjects that are captured in six
different sessions reflecting real-life scenarios of smartphone assisted
authentication. One of the unique features of this dataset is that it is
collected in four different geographic locations representing a diverse
population and ethnicity. Additionally, it also contains a multimodal
Presentation Attack (PA) or spoofing dataset using low-cost Presentation Attack
Instruments (PAI) such as print and electronic display attacks. The novel
acquisition protocols and the diversity of the data subjects collected from
different geographic locations will allow developing a novel algorithm for
either unimodal or multimodal biometrics.

PAD protocols are created according to the SWAN-PAD-protocols document.
Bona-fide session 2 data is split into 3 sets of training, development, and
evaluation. The bona-fide data from sessions 3,4,5,6 are used for evaluation as
well. PA samples are randomly split into 3 sets of training, development, and
evaluation. All the random splits are done 10 times to created 10 different
protocols. The PAD protocols contain only one type of attacks. For convenience,
PA_F and PA_V protocols are created for face and voice, respectively which
contain all the attacks.
"""
from bob.pad.face.database import SwanPadDatabase

database = SwanPadDatabase()
