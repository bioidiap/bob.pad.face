#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST
#
# Copyright (C) Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This file contains the python (distutils/setuptools) instructions so your
# package can be installed on **any** host system. It defines some basic
# information like the package name for instance, or its homepage.
#
# It also defines which other packages this python package depends on and that
# are required for this package's operation. The python subsystem will make
# sure all dependent packages are installed or will install them for you upon
# the installation of this package.
#
# The 'buildout' system we use here will go further and wrap this package in
# such a way to create an isolated python working environment. Buildout will
# make sure that dependencies which are not yet installed do get installed, but
# **without** requiring administrative privileges on the host system. This
# allows you to test your package with new python dependencies w/o requiring
# administrative interventions.

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
    author = 'Amir Mohammadi',
    author_email = 'amir.mohammadi@idiap.ch',
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
            'replay = bob.pad.face.config.database.replay:database',
            ],

        # registered preprocessors:
        'bob.pad.preprocessor': [
            'video-face-crop-preproc-64 = bob.pad.face.config.preprocessor.video_face_crop:video_face_crop_preproc_64_64',
            ],

        # registered preprocessors:
        'bob.pad.extractor': [
            'video-lbp-histogram-extractor-n8r1-uniform = bob.pad.face.config.extractor.video_lbp_histogram:video_lbp_histogram_extractor_n8r1_uniform',
            ],

        # registered algorithms:
        'bob.pad.algorithm': [
            'video-svm-pad-algorithm-10k-grid = bob.pad.face.config.algorithm.video_svm_pad_algorithm:video_svm_pad_algorithm_10k_grid',
            'video-svm-pad-algorithm-10k-grid-mean-std = bob.pad.face.config.algorithm.video_svm_pad_algorithm:video_svm_pad_algorithm_10k_grid_mean_std',
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
