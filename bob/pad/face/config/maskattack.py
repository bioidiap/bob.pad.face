from bob.pad.face.database import MaskAttackPadDatabase
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector
from bob.extension import rc

database = DatabaseConnector(
    MaskAttackPadDatabase(
        protocol="classification",
        original_directory=rc.get("bob.db.maskattack.directory"),
        original_extension=".avi",
    )
)
