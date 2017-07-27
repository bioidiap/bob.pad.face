#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import setup, dist
dist.Distribution(dict(setup_requires = ['bob.extension']))

# load the requirements.txt for additional requirements
from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name = 'bob.pad.face',
    version = open("version.txt").read().rstrip(),
    description = 'Implements tools for spoofing or presentation attack detection in face biometrics',

    url = 'https://gitlab.idiap.ch/bob/bob.pad.face',
    license = 'GPLv3',
    author = 'Olegs Nikisins',
    author_email = 'olegs.nikisins@idiap.ch',
    keywords = 'bob',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description = open('README.rst').read(),

    # This line is required for any distutils based packaging.
    # It will find all package-data inside the 'bob' directory.
    packages = find_packages('bob'),
    include_package_data = True,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires = install_requires,

    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    entry_points = {

        # scripts should be declared using this entry:
        'console_scripts' : [
            'version.py = bob.pad.face.script.version:main',
            ],

        # registered databases:
        'bob.pad.database': [
            'replay-attack = bob.pad.face.config.database.replay_attack:database',
            'replay-mobile = bob.pad.face.config.database.replay_mobile:database',
            'msu-mfsd = bob.pad.face.config.database.msu_mfsd:database',
            'aggregated-db = bob.pad.face.config.database.aggregated_db:database',
            ],

        # registered configurations:
        'bob.bio.config': [
            # databases
            'replay-attack = bob.pad.face.config.replay_attack',
            'replay-mobile = bob.pad.face.config.replay_mobile',
            'msu-mfsd = bob.pad.face.config.msu_mfsd',
            'aggregated-db = bob.pad.face.config.aggregated_db',

            # baselines:
            'lbp-svm = bob.pad.face.config.lbp_svm',
            'lbp-svm-aggregated-db = bob.pad.face.config.lbp_svm_aggregated_db',
            'qm-svm = bob.pad.face.config.qm_svm',
            'qm-svm-aggregated-db = bob.pad.face.config.qm_svm_aggregated_db',
            'frame-diff-svm = bob.pad.face.config.frame_diff_svm',
            'frame-diff-svm-aggregated-db = bob.pad.face.config.frame_diff_svm_aggregated_db',
            ],

        # registered preprocessors:
        'bob.pad.preprocessor': [
            'empty-preprocessor = bob.pad.face.config.preprocessor.filename:empty_preprocessor', # no preprocessing
            ],

        # registered preprocessors:
        'bob.pad.extractor': [
            'video-lbp-histogram-extractor-n8r1-uniform = bob.pad.face.config.extractor.video_lbp_histogram:video_lbp_histogram_extractor_n8r1_uniform',
            'video-quality-measure-galbally-msu = bob.pad.face.config.extractor.video_quality_measure:video_quality_measure_galbally_msu',
            'frame-diff-feat-extr-w20-over0 = bob.pad.face.config.extractor.frame_diff_features:frame_diff_feat_extr_w20_over0',
            ],

        # registered algorithms:
        'bob.pad.algorithm': [
            'video-svm-pad-algorithm-10k-grid-mean-std = bob.pad.face.config.algorithm.video_svm_pad_algorithm:video_svm_pad_algorithm_10k_grid_mean_std',
            'video-svm-pad-algorithm-10k-grid-mean-std-frame-level = bob.pad.face.config.algorithm.video_svm_pad_algorithm:video_svm_pad_algorithm_10k_grid_mean_std_frame_level',
            'video-svm-pad-algorithm-default-svm-param-mean-std-frame-level = bob.pad.face.config.algorithm.video_svm_pad_algorithm:video_svm_pad_algorithm_default_svm_param_mean_std_frame_level',
            ],

        # registered grid configurations:
        'bob.pad.grid': [
            'idiap = bob.pad.face.config.grid:idiap',
            'idiap-user-machines = bob.pad.face.config.grid:idiap_user_machines',
            ],

    },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
