"""`Replay-Mobile`_ is a database for face PAD experiments.

The Replay-Mobile Database for face spoofing consists of 1030 video clips of photo and video attack attempts to 40 clients,
under different lighting conditions.
These videos were recorded with current devices from the market -- an iPad Mini2 (running iOS) and a LG-G4 smartphone (running Android).
This Database was produced at the Idiap Research Institute (Switzerland) within the framework
of collaboration with Galician Research and Development Center in Advanced Telecommunications - Gradiant (Spain).
The reference citation is [CBVM16]_.

You can download the raw data of the `Replay-Mobile`_ database by following
the link.

.. include:: links.rst
"""
from bob.pad.face.database import ReplayMobilePadDatabase

database = ReplayMobilePadDatabase()
