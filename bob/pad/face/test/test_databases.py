#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

from unittest import SkipTest

import numpy as np

import bob.bio.base


def test_replay_attack():
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
        len(database.samples(groups=["train", "dev", "eval"], purposes="real"))
        == 200
    )
    assert (
        len(
            database.samples(groups=["train", "dev", "eval"], purposes="attack")
        )
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
        np.testing.assert_equal(sample.data[0][:, 0, 0], [8, 9, 11])
    except (RuntimeError, FileNotFoundError) as e:
        raise SkipTest(e)


def test_replay_mobile():
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
        len(database.samples(groups=["train", "dev", "eval"], purposes="real"))
        == 390
    )
    assert (
        len(
            database.samples(groups=["train", "dev", "eval"], purposes="attack")
        )
        == 640
    )

    all_samples = database.sort(database.samples())
    sample = all_samples[0]
    assert (
        sample.key
        == "devel/attack/attack_client005_session01_mattescreen_fixed_mobile_photo_lightoff.mov"
    ), sample.key
    assert sample.should_flip
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
    }, annot

    sample2 = [s for s in all_samples if not s.should_flip][0]
    assert (
        sample2.key
        == "devel/attack/attack_client005_session01_mattescreen_fixed_tablet_photo_lightoff.mov"
    ), sample2.key
    assert not sample2.should_flip
    annot = dict(sample2.annotations["0"])
    assert annot["leye"][1] > annot["reye"][1], annot
    assert annot == {
        "reye": [873, 305],
        "leye": [879, 423],
        "nose": [937, 365],
        "mouthleft": [1018, 313],
        "mouthright": [1023, 405],
        "topleft": [747, 226],
        "bottomright": [1111, 495],
    }, annot

    try:
        assert sample.data.shape == (20, 3, 1280, 720), sample.data.shape
        np.testing.assert_equal(sample.data[0][:, 0, 0], [13, 13, 13])
        assert sample2.data.shape == (20, 3, 1280, 720), sample2.data.shape
        np.testing.assert_equal(sample2.data[0][:, 0, 0], [19, 33, 30])
    except (RuntimeError, FileNotFoundError) as e:
        raise SkipTest(e)


# Test the mask_attack database
def test_mask_attack():
    mask_attack = bob.bio.base.load_resource(
        "mask-attack",
        "database",
        preferred_package="bob.pad.face",
        package_prefix="bob.pad.",
    )
    # all real sequences: 2 sessions, 5 recordings for 17 individuals
    assert (
        len(
            mask_attack.samples(
                groups=["train", "dev", "eval"], purposes="real"
            )
        )
        == 170
    )
    # all attacks: 1 session, 5 recordings for 17 individuals
    assert (
        len(
            mask_attack.samples(
                groups=["train", "dev", "eval"], purposes="attack"
            )
        )
        == 85
    )

    # training real: 7 subjects, 2 sessions, 5 recordings
    assert len(mask_attack.samples(groups=["train"], purposes="real")) == 70
    # training real: 7 subjects, 1 session, 5 recordings
    assert len(mask_attack.samples(groups=["train"], purposes="attack")) == 35

    # dev and test contains the same number of sequences:
    # real: 5 subjects, 2 sessions, 5 recordings
    # attack: 5 subjects, 1 sessions, 5 recordings
    assert len(mask_attack.samples(groups=["dev"], purposes="real")) == 50
    assert len(mask_attack.samples(groups=["eval"], purposes="real")) == 50
    assert len(mask_attack.samples(groups=["dev"], purposes="attack")) == 25
    assert len(mask_attack.samples(groups=["eval"], purposes="attack")) == 25

    sample = mask_attack.samples()[0]
    try:
        assert sample.data.shape == (20, 3, 480, 640)
        np.testing.assert_equal(sample.data[0][:, 0, 0], [185, 166, 167])
        annot = sample.annotations["0"]
        assert annot["leye"][1] > annot["reye"][1], annot
        assert annot == {
            "leye": [212, 287],
            "reye": [217, 249],
        }
        assert sample.depth.shape == (20, 480, 640)
    except FileNotFoundError as e:
        raise SkipTest(e)


def test_casia_fasd():
    casia_fasd = bob.bio.base.load_resource(
        "casia-fasd",
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
    sample = casia_fasd.samples()[0]
    try:
        assert sample.data.shape == (20, 3, 480, 640)
        np.testing.assert_equal(sample.data[0][:, 0, 0], [217, 228, 227])
    except FileNotFoundError as e:
        raise SkipTest(e)


def test_casia_surf():
    try:
        casia_surf = bob.bio.base.load_resource(
            "casia-surf",
            "database",
            preferred_package="bob.pad.face",
            package_prefix="bob.pad.",
        )

        assert len(casia_surf.samples()) == 96584
        assert len(casia_surf.samples(purposes="real")) == 29394
        assert len(casia_surf.samples(purposes="attack")) == 67190
        assert len(casia_surf.samples(groups=("train", "dev"))) == 38874
        assert len(casia_surf.samples(groups="train")) == 29266
        assert len(casia_surf.samples(groups="dev")) == 9608
        assert len(casia_surf.samples(groups="eval")) == 57710
        sample = casia_surf.samples()[0]
        assert sample.data.shape == (1, 3, 279, 279)
        np.testing.assert_equal(sample.data[0][:, 0, 0], [0, 0, 0])
        assert sample.depth.shape == (1, 143, 143)
        assert sample.infrared.shape == (1, 143, 143)
    except FileNotFoundError as e:
        raise SkipTest(e)


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
        len(database.samples(groups=["train", "dev", "eval"], purposes="real"))
        == 3300
    )
    assert (
        len(
            database.samples(groups=["train", "dev", "eval"], purposes="attack")
        )
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
        assert sample.data.shape == (20, 3, 1280, 720)
        np.testing.assert_equal(sample.data[0][:, 0, 0], [255, 255, 253])
    except (RuntimeError, FileNotFoundError) as e:
        raise SkipTest(e)


def test_oulu_npu():
    database = bob.bio.base.load_resource(
        "oulu-npu",
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
    assert (
        len(database.samples(groups=["train", "dev", "eval"]))
        == 1200 + 900 + 600
    )
    assert len(database.samples(groups=["train", "dev"])) == 1200 + 900
    assert len(database.samples(groups=["train"])) == 1200
    assert (
        len(database.samples(groups=["train", "dev", "eval"], purposes="real"))
        == 240 + 180 + 120
    )
    assert (
        len(
            database.samples(groups=["train", "dev", "eval"], purposes="attack")
        )
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
        np.testing.assert_equal(sample.data[0][:, 0, 0], [195, 191, 199])
    except (RuntimeError, FileNotFoundError) as e:
        raise SkipTest(e)
