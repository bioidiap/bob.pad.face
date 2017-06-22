#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.base.test.utils import db_available

@db_available('replay')
def test_replay():
    replay_database_instance = bob.bio.base.load_resource('replay', 'database', preferred_package='bob.pad.face', package_prefix='bob.pad.')
    try:

        assert len( replay_database_instance.objects(groups=['train', 'dev', 'eval']) )==  1200
        assert len( replay_database_instance.objects(groups=['train', 'dev']) ) ==  720
        assert len( replay_database_instance.objects(groups=['train']) ) ==  360
        assert len( replay_database_instance.objects(groups=['train', 'dev', 'eval'], protocol = 'grandtest') )==  1200
        assert len( replay_database_instance.objects(groups=['train', 'dev', 'eval'], protocol = 'grandtest', purposes='real') ) ==  200
        assert len( replay_database_instance.objects(groups=['train', 'dev', 'eval'], protocol = 'grandtest', purposes='attack') ) == 1000

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)


@db_available('replaymobile')
def test_replaymobile():
    replaymobile = bob.bio.base.load_resource('replay-mobile', 'database', preferred_package='bob.pad.face', package_prefix='bob.pad.')
    try:

        assert len( replaymobile.objects(groups=['train', 'dev', 'eval']) )==  1030
        assert len( replaymobile.objects(groups=['train', 'dev']) ) ==  728
        assert len( replaymobile.objects(groups=['train']) ) ==  312
        assert len( replaymobile.objects(groups=['train', 'dev', 'eval'], protocol = 'grandtest') )==  1030
        assert len( replaymobile.objects(groups=['train', 'dev', 'eval'], protocol = 'grandtest', purposes='real') ) ==  390
        assert len( replaymobile.objects(groups=['train', 'dev', 'eval'], protocol = 'grandtest', purposes='attack') ) == 640

    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)