from bob.pad.face.database import ReplayPadDatabase
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector
from bob.extension import rc

database = DatabaseConnector(
    ReplayPadDatabase(
        protocol="grandtest",
        original_directory=rc.get("bob.db.replay.directory"),
        original_extension=".mov",
    )
)
