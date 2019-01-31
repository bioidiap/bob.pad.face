#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.base.test.utils import db_available


@db_available('replay')  # the name of the package
def test_replay():
    replay_database_instance = bob.bio.base.load_resource(
        'replay-attack',
        'database',
        preferred_package='bob.pad.face',
        package_prefix='bob.pad.'
    )  # replay-attack is the name of the configuration file
    try:

        assert len(
            replay_database_instance.objects(
                groups=['train', 'dev', 'eval'])) == 1200
        assert len(
            replay_database_instance.objects(groups=['train', 'dev'])) == 720
        assert len(replay_database_instance.objects(groups=['train'])) == 360
        assert len(
            replay_database_instance.objects(
                groups=['train', 'dev', 'eval'], protocol='grandtest')) == 1200
        assert len(
            replay_database_instance.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='real')) == 200
        assert len(
            replay_database_instance.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='attack')) == 1000

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e)


@db_available('replaymobile')
def test_replaymobile():
    replaymobile = bob.bio.base.load_resource(
        'replay-mobile',
        'database',
        preferred_package='bob.pad.face',
        package_prefix='bob.pad.')
    try:

        assert len(
            replaymobile.objects(groups=['train', 'dev', 'eval'])) == 1030
        assert len(replaymobile.objects(groups=['train', 'dev'])) == 728
        assert len(replaymobile.objects(groups=['train'])) == 312
        assert len(
            replaymobile.objects(
                groups=['train', 'dev', 'eval'], protocol='grandtest')) == 1030
        assert len(
            replaymobile.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='real')) == 390
        assert len(
            replaymobile.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='attack')) == 640

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e)


@db_available('msu_mfsd_mod')
def test_msu_mfsd():
    msu_mfsd = bob.bio.base.load_resource(
        'msu-mfsd',
        'database',
        preferred_package='bob.pad.face',
        package_prefix='bob.pad.')
    try:

        assert len(msu_mfsd.objects(groups=['train', 'dev', 'eval'])) == 280
        assert len(msu_mfsd.objects(groups=['train', 'dev'])) == 160
        assert len(msu_mfsd.objects(groups=['train'])) == 80
        assert len(
            msu_mfsd.objects(
                groups=['train', 'dev', 'eval'], protocol='grandtest')) == 280
        assert len(
            msu_mfsd.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='real')) == 70
        assert len(
            msu_mfsd.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='attack')) == 210

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e)


# Test the maskattack database
@db_available('maskattack')
def test_maskattack():
    maskattack = bob.bio.base.load_resource(
        'maskattack',
        'database',
        preferred_package='bob.pad.face',
        package_prefix='bob.pad.')
    try:
        # all real sequences: 2 sessions, 5 recordings for 17 individuals
        assert len(maskattack.objects(groups=['train', 'dev', 'eval'], purposes='real')) == 170
        # all attacks: 1 session, 5 recordings for 17 individuals
        assert len(maskattack.objects(groups=['train', 'dev', 'eval'], purposes='attack')) == 85
        
        # training real: 7 subjects, 2 sessions, 5 recordings
        assert len(maskattack.objects(groups=['train'], purposes='real')) == 70
        # training real: 7 subjects, 1 session, 5 recordings
        assert len(maskattack.objects(groups=['train'], purposes='attack')) == 35

        # dev and test contains the same number of sequences:
        # real: 5 subjects, 2 sessions, 5 recordings
        # attack: 5 subjects, 1 sessions, 5 recordings
        assert len(maskattack.objects(groups=['dev'], purposes='real')) == 50
        assert len(maskattack.objects(groups=['eval'], purposes='real')) == 50
        assert len(maskattack.objects(groups=['dev'], purposes='attack')) == 25
        assert len(maskattack.objects(groups=['eval'], purposes='attack')) == 25

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e)

# Test the Aggregated database, which doesn't have a package
def test_aggregated_db():
    aggregated_db = bob.bio.base.load_resource(
        'aggregated-db',
        'database',
        preferred_package='bob.pad.face',
        package_prefix='bob.pad.')
    try:

        assert len(
            aggregated_db.objects(groups=['train', 'dev', 'eval'])) == 2510
        assert len(aggregated_db.objects(groups=['train', 'dev'])) == 1608
        assert len(aggregated_db.objects(groups=['train'])) == 752

        assert len(aggregated_db.objects(groups='train')) == 752
        assert len(aggregated_db.objects(groups='dev')) == 856
        assert len(aggregated_db.objects(groups='eval')) == 902

        assert len(
            aggregated_db.objects(
                groups=['train', 'dev', 'eval'], protocol='grandtest')) == 2510
        assert len(
            aggregated_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='real')) == 660
        assert len(
            aggregated_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='attack')) == 1850

        assert len(
            aggregated_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='photo-photo-video')) == 1664
        assert len(
            aggregated_db.objects(
                groups=['train', 'dev'], protocol='photo-photo-video')) == 1176
        assert len(
            aggregated_db.objects(groups='eval',
                                  protocol='photo-photo-video')) == 488

        assert len(
            aggregated_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='video-video-photo')) == 1506
        assert len(
            aggregated_db.objects(
                groups=['train', 'dev'], protocol='video-video-photo')) == 872
        assert len(
            aggregated_db.objects(groups='eval',
                                  protocol='video-video-photo')) == 634

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e)


# Test the casiasurf database
@db_available('casiasurf')
def test_casiasurf():
    casiasurf = bob.bio.base.load_resource(
        'casiasurf',
        'database',
        preferred_package='bob.pad.face',
        package_prefix='bob.pad.')
    try:
        assert len(casiasurf.objects(groups=['train', 'dev'], purposes='real')) == 8942 
        assert len(casiasurf.objects(groups=['train'], purposes='attack')) == 20324
        assert len(casiasurf.objects(groups=['dev'], purposes='real')) == 0
        assert len(casiasurf.objects(groups=['dev'], purposes='attack')) == 9608 
        
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e)

# Test the BATL database
def test_aggregated_db():
    batl_db = bob.bio.base.load_resource(
        'batl-db',
        'database',
        preferred_package='bob.pad.face',
        package_prefix='bob.pad.')
    try:

        assert len(
            batl_db.objects(groups=['train', 'dev', 'eval'])) == 1679
        assert len(batl_db.objects(groups=['train', 'dev'])) == 1122
        assert len(batl_db.objects(groups=['train'])) == 565

        assert len(batl_db.objects(groups='train')) == 565
        assert len(batl_db.objects(groups='dev')) == 557
        assert len(batl_db.objects(groups='eval')) == 557

        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'], protocol='grandtest')) == 1679
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='real')) == 347
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest',
                purposes='attack')) == 1332
        #tests for join_train_dev protocols
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-join_train_dev')) == 1679
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-join_train_dev')) == 1679
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-join_train_dev')) == 557
        # test for LOO_fakehead
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-LOO_fakehead')) == 1149
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-LOO_fakehead')) == 1017
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-LOO_fakehead')) == 132

        # test for LOO_flexiblemask
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-LOO_flexiblemask')) == 1132
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-LOO_flexiblemask')) == 880
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-LOO_flexiblemask')) == 252

        # test for LOO_glasses
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-LOO_glasses')) == 1206
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-LOO_glasses')) == 1069
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-LOO_glasses')) == 137

        # test for LOO_papermask
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-LOO_papermask')) == 1308
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-LOO_papermask')) == 1122
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-LOO_papermask')) == 186

        # test for LOO_prints
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-LOO_prints')) == 1169
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-LOO_prints')) == 988
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-LOO_prints')) == 181

        # test for LOO_replay
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-LOO_replay')) == 1049
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-LOO_replay')) == 854
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-LOO_replay')) == 195

        # test for LOO_rigidmask
        assert len(
            batl_db.objects(
                groups=['train', 'dev', 'eval'],
                protocol='grandtest-color-50-LOO_rigidmask')) == 1198
        assert len(
            batl_db.objects(
                groups=['train', 'dev'], protocol='grandtest-color-50-LOO_rigidmask')) == 1034
        assert len(
            batl_db.objects(groups='eval',
                                  protocol='grandtest-color-50-LOO_rigidmask')) == 164


    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'"
            % e)