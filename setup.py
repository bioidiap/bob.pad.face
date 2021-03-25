#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import setup, dist

dist.Distribution(dict(setup_requires=["bob.extension"]))

# load the requirements.txt for additional requirements
from bob.extension.utils import load_requirements, find_packages

install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(
    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name="bob.pad.face",
    version=open("version.txt").read().rstrip(),
    description="Implements tools for spoofing or presentation attack detection in face biometrics",
    url="https://gitlab.idiap.ch/bob/bob.pad.face",
    license="GPLv3",
    author="Amir Mohammadi",
    author_email="amir.mohammadi@idiap.ch",
    keywords="bob",
    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open("README.rst").read(),
    # This line is required for any distutils based packaging.
    # It will find all package-data inside the 'bob' directory.
    packages=find_packages("bob"),
    include_package_data=True,
    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires=install_requires,
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
    entry_points={
        # scripts should be declared using this entry:
        "console_scripts": [
        ],
        # registered databases:
        "bob.pad.database": [
            "replay-attack = bob.pad.face.config.replay_attack:database",
            "replay-mobile = bob.pad.face.config.replay_mobile:database",
            "casiafasd = bob.pad.face.config.casiafasd:database",
            "maskattack = bob.pad.face.config.maskattack:database",
            "casiasurf-color = bob.pad.face.config.casiasurf_color:database",
            "casiasurf = bob.pad.face.config.casiasurf:database",
            "swan = bob.pad.face.config.swan:database",
            "oulunpu = bob.pad.face.config.oulunpu:database",
        ],
        # registered configurations:
        "bob.pad.config": [
            # databases
            "replay-attack = bob.pad.face.config.replay_attack",
            "replay-mobile = bob.pad.face.config.replay_mobile",
            "casiafasd = bob.pad.face.config.casiafasd",
            "maskattack = bob.pad.face.config.maskattack",
            "casiasurf-color = bob.pad.face.config.casiasurf_color",
            "casiasurf = bob.pad.face.config.casiasurf",
            "swan = bob.pad.face.config.swan",
            "oulunpu = bob.pad.face.config.oulunpu",
            # LBPs
            "lbp = bob.pad.face.config.lbp_64",
            # quality measure
            "qm = bob.pad.face.config.qm_64",
            # classifiers
            "svm-frames = bob.pad.face.config.svm_frames",
        ],
        # registered ``bob pad ...`` commands
        "bob.pad.cli": [
            "statistics        = bob.pad.face.script.statistics:statistics",
        ],
    },
    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
