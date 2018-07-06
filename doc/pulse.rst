
.. _bob.pad.face.pulse:

===============
Pulse-based PAD
===============

In this section, we briefly describe our work made for face
presentation attack detection using the blood volume pulse,
inferred from remote photoplesthymograpy.

The basic idea here is to retrieve the pulse signals from
face video sequences, to derive features from their frequency
spectrum and then to learn a classifier to discriminate
between *bonafide* attempts from presentation attacks.

For this purpose, we describe both :py:class:`bob.bio.base.preprocessor.Preprocessor` and
:py:class:`bob.bio.base.extractor.Extractor` specifically dedicated to this task.

Preprocessors: Pulse Extraction
-------------------------------

Preprocessors basically extract pulse signals from face video 
sequences. They heavily rely on what has been done in `bob.rppg.base`
so you may want to have a look at `its documentation <https://www.idiap.ch/software/bob/docs/bob/bob.rppg.base/master/index.html>`_. 

In this package, 4 preprocessors have been implemented:

  1. :py:class:`bob.pad.face.preprocessor.LiPulseExtraction` described in [Li_ICPR_2016]_.
  
  2. :py:class:`bob.pad.face.preprocessor.Chrom` described in [CHROM]_.

  3. :py:class:`bob.pad.face.preprocessor.SSR` described in [SSR]_.

  4. :py:class:`bob.pad.face.preprocessor.PPGSecure` described in [NOWARA]_.


Extractors: Features from Pulses
--------------------------------

Extractors compute and retrieve features from the pulse signal. All
implemented extractors act on the frequency spectrum of the pulse signal.

In this package, 3 extractors have been implemented:

  1. :py:class:`bob.pad.face.extractor.LiSpectralFeatures` described in [Li_ICPR_2016]_.
  
  2. :py:class:`bob.pad.face.extractor.PPGSecure` described in [NOWARA]_.

  3. :py:class:`bob.pad.face.extractor.LTSS` described in [LTSS]_.



References
----------


.. [Li_ICPR_2016] *X. Li, J, Komulainen, G. Zhao, P-C Yuen and M. Pietikäinen*
  **Generalized face anti-spoofing by detecting pulse from face videos**,
  Intl Conf on Pattern Recognition (ICPR), 2016

.. [CHROM] *de Haan, G. & Jeanne, V*. **Robust Pulse Rate from Chrominance based rPPG**, IEEE Transactions on Biomedical Engineering, 2013. `pdf <http://www.es.ele.tue.nl/~dehaan/pdf/169_ChrominanceBasedPPG.pdf>`__

.. [SSR] *Wang, W., Stuijk, S. and de Haan, G*. **A Novel Algorithm for Remote Photoplesthymograpy: Spatial Subspace Rotation**, IEEE Trans. On Biomedical Engineering, 2015

.. [NOWARA] *E. M. Nowara, A. Sabharwal, A. Veeraraghavan*. **PPGSecure: Biometric Presentation Attack Detection Using Photopletysmograms**, IEEE International Conference on Automatic Face & Gesture Recognition, 2017

.. [LTSS] *H .Muckenhirn, P. Korshunov, M. Magimai-Doss, S Marcel*. **Long-Term Spectral Statistics for Voice Presentation Attack Detection**, IEEE Trans. On Audio, Speech and Language Processing, 2017
