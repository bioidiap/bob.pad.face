
.. _bob.pad.face.installation:

======================
 Setting up Databases
======================

To run face PAD algorithms using this package, you'll need to make sure to
download the raw files corresponding to the databases you'd like to process.
The raw files are **not** distributed with Bob_ software as biometric data is,
to most countries, considered sensible data that cannot be obtained without
explicit licensing from a data controller. You must visit the websites below,
sign the license agreements and then download the data before trying out to run
the baselines.

.. note::

   If you're at the Idiap Research Institute in Switzerland, the datasets in
   the baselines mentioned in this guide are already downloaded and
   pre-installed on our shared file system. You don't need to re-download
   databases.


The current system readily supports the following freely available datasets:

* `REPLAY-ATTACK`_
* `REPLAY-MOBILE`_
* `SWAN`_
* `OULU-NPU`_
* `MASK-ATTACK`_

After downloading the databases, annotate the base directories in which they
are installed. Then, follow the instructions in
:ref:`bob.pad.base.installation` to let this framework know where databases are
located on your system.


.. include:: links.rst
