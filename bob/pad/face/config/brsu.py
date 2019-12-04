#!/usr/bin/env python
# encoding: utf-8

from bob.pad.face.database import BRSUPadDatabase 
from bob.extension import rc

database = BRSUPadDatabase(
    protocol='test',
    original_directory=rc['bob.db.brsu.directory'],
)
