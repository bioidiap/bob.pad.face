#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.base.test.utils import db_available


def test_replayattack():
    database = bob.bio.base.load_resource(
        "replay-attack",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )

    assert database.protocols() == ['digitalphoto', 'grandtest', 'highdef', 'mobile', 'photo', 'print', 'smalltest', 'video']
    assert database.groups() == ["dev", "eval", "train"]
    assert len(database.samples(groups=["train", "dev", "eval"])) == 1200
    assert len(database.samples(groups=["train", "dev"])) == 720
    assert len(database.samples(groups=["train"])) == 360
    assert len(database.samples(groups=["train", "dev", "eval"])) == 1200
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="real")) == 200
    )
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="attack")) == 1000
    )

    sample = database.sort(database.samples())[0]
    try:
        assert sample.annotations["0"] == {
            "bottomright": [213, 219],
            "topleft": [58, 108],
            "leye": [118, 190],
            "reye": [117, 137],
            "mouthleft": [177, 144],
            "mouthright": [180, 183],
            "nose": [152, 164],
        }
        assert sample.data.shape == (20, 3, 240, 320)
        assert sample.data[0][0, 0, 0] == 8
    except IOError as e:
        raise SkipTest(e)



def test_replaymobile():
    database = bob.bio.base.load_resource(
        "replay-mobile",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )

    assert database.protocols() == ["grandtest", "mattescreen", "print"]
    assert database.groups() == ["dev", "eval", "train"]
    assert len(database.samples(groups=["train", "dev", "eval"])) == 1030
    assert len(database.samples(groups=["train", "dev"])) == 728
    assert len(database.samples(groups=["train"])) == 312
    assert len(database.samples(groups=["train", "dev", "eval"])) == 1030
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="real")) == 390
    )
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="attack")) == 640
    )

    sample = database.sort(database.samples())[0]
    try:
        assert sample.annotations["0"] == {
            "bottomright": [760, 498],
            "topleft": [374, 209],
            "leye": [518, 417],
            "reye": [522, 291],
            "mouthleft": [669, 308],
            "mouthright": [666, 407],
            "nose": [585, 358],
        }
        assert sample.data.shape == (20, 3, 720, 1280)
        assert sample.data[0][0, 0, 0] == 13
    except IOError as e:
        raise SkipTest(e)


# Test the maskattack database
@db_available("maskattack")
def test_maskattack():
    maskattack = bob.bio.base.load_resource(
        "maskattack",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    ).database
    try:
        # all real sequences: 2 sessions, 5 recordings for 17 individuals
        assert (
            len(maskattack.objects(groups=["train", "dev", "eval"], purposes="real"))
            == 170
        )
        # all attacks: 1 session, 5 recordings for 17 individuals
        assert (
            len(maskattack.objects(groups=["train", "dev", "eval"], purposes="attack"))
            == 85
        )

        # training real: 7 subjects, 2 sessions, 5 recordings
        assert len(maskattack.objects(groups=["train"], purposes="real")) == 70
        # training real: 7 subjects, 1 session, 5 recordings
        assert len(maskattack.objects(groups=["train"], purposes="attack")) == 35

        # dev and test contains the same number of sequences:
        # real: 5 subjects, 2 sessions, 5 recordings
        # attack: 5 subjects, 1 sessions, 5 recordings
        assert len(maskattack.objects(groups=["dev"], purposes="real")) == 50
        assert len(maskattack.objects(groups=["eval"], purposes="real")) == 50
        assert len(maskattack.objects(groups=["dev"], purposes="attack")) == 25
        assert len(maskattack.objects(groups=["eval"], purposes="attack")) == 25

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )


# Test the casiasurf database
@db_available("casiasurf")
def test_casiasurf():
    casiasurf = bob.bio.base.load_resource(
        "casiasurf",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    ).database
    try:
        assert len(casiasurf.objects(groups=["train"], purposes="real")) == 8942
        assert len(casiasurf.objects(groups=["train"], purposes="attack")) == 20324
        assert len(casiasurf.objects(groups=("dev",), purposes=("real",))) == 2994
        assert len(casiasurf.objects(groups=("dev",), purposes=("attack",))) == 6614
        assert (
            len(casiasurf.objects(groups=("dev",), purposes=("real", "attack"))) == 9608
        )
        assert len(casiasurf.objects(groups=("eval",), purposes=("real",))) == 17458
        assert len(casiasurf.objects(groups=("eval",), purposes=("attack",))) == 40252
        assert (
            len(casiasurf.objects(groups=("eval",), purposes=("real", "attack")))
            == 57710
        )

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e
        )


@db_available("casia_fasd")
def test_casia_fasd():
    casia_fasd = bob.bio.base.load_resource(
        "casiafasd",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    ).database

    assert len(casia_fasd.objects()) == 600
    assert len(casia_fasd.objects(purposes="real")) == 150
    assert len(casia_fasd.objects(purposes="attack")) == 450
    assert len(casia_fasd.objects(groups=("train", "dev"))) == 240
    assert len(casia_fasd.objects(groups="train")) == 180
    assert len(casia_fasd.objects(groups="dev")) == 60
    assert len(casia_fasd.objects(groups="eval")) == 360

    # test annotations since they are shipped with bob.db.casia_fasd
    f = [f for f in casia_fasd.objects() if f.path == "train_release/1/2"][0]
    assert len(f.annotations) == 132
    a = f.annotations["0"]
    oracle = {
        "topleft": (102, 214),
        "bottomright": (242, 354),
        "reye": (151.0, 249.0),
        "leye": (151.0, 319.0),
    }
    assert a == oracle, a


@db_available("casia_fasd")
def test_casia_fasd_frames():
    casia_fasd = bob.bio.base.load_resource(
        "casiafasd",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    ).database

    # test frame loading if the db original files are available
    try:
        files = casia_fasd.objects()[:12]
        for f in files:
            for frame in f.frames:
                assert frame.shape == (3, 1280, 720)
                break
    except (IOError, RuntimeError) as e:
        raise SkipTest(
            "The database original files are missing. To run this test run "
            "``bob config set bob.db.casia_fasd.directory "
            "/path/to/casia_fasd_files`` in a terminal to point to the "
            "original files on your computer. . Here is the error: '%s'" % e
        )
