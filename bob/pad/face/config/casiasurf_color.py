from bob.pad.face.database import CasiaSurfPadDatabase
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector
from bob.extension import rc

database = DatabaseConnector(
    CasiaSurfPadDatabase(
        protocol="color",
        original_directory=rc.get("bob.db.casiasurf.directory"),
        original_extension=".jpg",
    )
)
