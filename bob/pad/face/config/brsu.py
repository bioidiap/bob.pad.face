from bob.pad.face.database import BRSUPadDatabase
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector
from bob.extension import rc

database = DatabaseConnector(
    BRSUPadDatabase(
        protocol="test",
        original_directory=rc["bob.db.brsu.directory"],
    )
)
