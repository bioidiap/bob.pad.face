#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

from nose.plugins.skip import SkipTest

import bob.bio.base


def test_replayattack():
    database = bob.bio.base.load_resource(
        "replay-attack",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )

    assert database.protocols() == [
        "digitalphoto",
        "grandtest",
        "highdef",
        "mobile",
        "photo",
        "print",
        "smalltest",
        "video",
    ]
    assert database.groups() == ["dev", "eval", "train"]
    assert len(database.samples(groups=["train", "dev", "eval"])) == 1200
    assert len(database.samples(groups=["train", "dev"])) == 720
    assert len(database.samples(groups=["train"])) == 360
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="real")) == 200
    )
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="attack"))
        == 1000
    )

    sample = database.sort(database.samples())[0]
    try:
        annot = dict(sample.annotations["0"])
        assert annot["leye"][1] > annot["reye"][1], annot
        assert annot == {
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
    except RuntimeError as e:
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
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="real")) == 390
    )
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="attack")) == 640
    )

    sample = database.sort(database.samples())[0]
    try:
        annot = dict(sample.annotations["0"])
        assert annot["leye"][1] > annot["reye"][1], annot
        assert annot == {
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
    except RuntimeError as e:
        raise SkipTest(e)


# Test the maskattack database
def test_maskattack():
    maskattack = bob.bio.base.load_resource(
        "maskattack",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )
    # all real sequences: 2 sessions, 5 recordings for 17 individuals
    assert (
        len(maskattack.samples(groups=["train", "dev", "eval"], purposes="real")) == 170
    )
    # all attacks: 1 session, 5 recordings for 17 individuals
    assert (
        len(maskattack.samples(groups=["train", "dev", "eval"], purposes="attack"))
        == 85
    )

    # training real: 7 subjects, 2 sessions, 5 recordings
    assert len(maskattack.samples(groups=["train"], purposes="real")) == 70
    # training real: 7 subjects, 1 session, 5 recordings
    assert len(maskattack.samples(groups=["train"], purposes="attack")) == 35

    # dev and test contains the same number of sequences:
    # real: 5 subjects, 2 sessions, 5 recordings
    # attack: 5 subjects, 1 sessions, 5 recordings
    assert len(maskattack.samples(groups=["dev"], purposes="real")) == 50
    assert len(maskattack.samples(groups=["eval"], purposes="real")) == 50
    assert len(maskattack.samples(groups=["dev"], purposes="attack")) == 25
    assert len(maskattack.samples(groups=["eval"], purposes="attack")) == 25


# Test the casiasurf database
# def test_casiasurf():
#     casiasurf = bob.bio.base.load_resource(
#         "casiasurf",
#         "database",
#         preferred_package="bob.pad.face",
#         package_prefix="bob.pad.",
#     )
#     assert len(casiasurf.samples(groups=["train"], purposes="real")) == 8942
#     assert len(casiasurf.samples(groups=["train"], purposes="attack")) == 20324
#     assert len(casiasurf.samples(groups=("dev",), purposes=("real",))) == 2994
#     assert len(casiasurf.samples(groups=("dev",), purposes=("attack",))) == 6614
#     assert (
#         len(casiasurf.samples(groups=("dev",), purposes=("real", "attack"))) == 9608
#     )
#     assert len(casiasurf.samples(groups=("eval",), purposes=("real",))) == 17458
#     assert len(casiasurf.samples(groups=("eval",), purposes=("attack",))) == 40252
#     assert (
#         len(casiasurf.samples(groups=("eval",), purposes=("real", "attack")))
#         == 57710
#     )


def test_casiasurf_color_protocol():
    casiasurf = bob.bio.base.load_resource(
        "casiasurf-color",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )
    assert len(casiasurf.samples(groups=["train"], purposes="real")) == 8942
    assert len(casiasurf.samples(groups=["train"], purposes="attack")) == 20324
    assert len(casiasurf.samples(groups=("dev",), purposes=("real",))) == 2994
    assert len(casiasurf.samples(groups=("dev",), purposes=("attack",))) == 6614
    assert len(casiasurf.samples(groups=("dev",), purposes=("real", "attack"))) == 9608
    assert len(casiasurf.samples(groups=("eval",), purposes=("real",))) == 17458
    assert len(casiasurf.samples(groups=("eval",), purposes=("attack",))) == 40252
    assert (
        len(casiasurf.samples(groups=("eval",), purposes=("real", "attack"))) == 57710
    )


def test_casia_fasd():
    casia_fasd = bob.bio.base.load_resource(
        "casiafasd",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )

    assert len(casia_fasd.samples()) == 600
    assert len(casia_fasd.samples(purposes="real")) == 150
    assert len(casia_fasd.samples(purposes="attack")) == 450
    assert len(casia_fasd.samples(groups=("train", "dev"))) == 240
    assert len(casia_fasd.samples(groups="train")) == 180
    assert len(casia_fasd.samples(groups="dev")) == 60
    assert len(casia_fasd.samples(groups="eval")) == 360


def test_swan():
    database = bob.bio.base.load_resource(
        "swan",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )

    assert database.protocols() == [
        "pad_p2_face_f1",
        "pad_p2_face_f2",
        "pad_p2_face_f3",
        "pad_p2_face_f4",
        "pad_p2_face_f5",
    ]
    assert database.groups() == ["dev", "eval", "train"]
    assert len(database.samples(groups=["train", "dev", "eval"])) == 5802
    assert len(database.samples(groups=["train", "dev"])) == 2803
    assert len(database.samples(groups=["train"])) == 2001
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="real")) == 3300
    )
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="attack"))
        == 2502
    )

    sample = database.sort(database.samples())[0]
    try:
        annot = dict(sample.annotations["0"])
        assert annot["leye"][1] > annot["reye"][1], annot
        assert annot == {
            "bottomright": [849, 564],
            "leye": [511, 453],
            "mouthleft": [709, 271],
            "mouthright": [711, 445],
            "nose": [590, 357],
            "reye": [510, 265],
            "topleft": [301, 169],
        }
        assert sample.data.shape == (20, 3, 720, 1280)
        assert sample.data[0][0, 0, 0] == 87
    except RuntimeError as e:
        raise SkipTest(e)


def test_oulunpu():
    database = bob.bio.base.load_resource(
        "oulunpu",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )

    assert database.protocols() == [
        "Protocol_1",
        "Protocol_1_2",
        "Protocol_1_3",
        "Protocol_2",
        "Protocol_3_1",
        "Protocol_3_2",
        "Protocol_3_3",
        "Protocol_3_4",
        "Protocol_3_5",
        "Protocol_3_6",
        "Protocol_4_1",
        "Protocol_4_2",
        "Protocol_4_3",
        "Protocol_4_4",
        "Protocol_4_5",
        "Protocol_4_6",
    ]
    assert database.groups() == ["dev", "eval", "train"]
    assert len(database.samples(groups=["train", "dev", "eval"])) == 1200 + 900 + 600
    assert len(database.samples(groups=["train", "dev"])) == 1200 + 900
    assert len(database.samples(groups=["train"])) == 1200
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="real"))
        == 240 + 180 + 120
    )
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="attack"))
        == 960 + 720 + 480
    )

    sample = database.sort(database.samples())[0]
    try:
        annot = dict(sample.annotations["0"])
        assert annot["leye"][1] > annot["reye"][1], annot
        assert annot == {
            "bottomright": [1124, 773],
            "leye": [818, 638],
            "mouthleft": [1005, 489],
            "mouthright": [1000, 634],
            "nose": [906, 546],
            "reye": [821, 470],
            "topleft": [632, 394],
        }
        assert sample.data.shape == (20, 3, 1920, 1080)
        assert sample.data[0][0, 0, 0] == 195
    except RuntimeError as e:
        raise SkipTest(e)
