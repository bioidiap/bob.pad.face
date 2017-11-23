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
