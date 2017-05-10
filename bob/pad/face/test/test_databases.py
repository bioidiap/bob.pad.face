#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Thu May 24 10:41:42 CEST 2012

from nose.plugins.skip import SkipTest

import bob.bio.base
from bob.bio.base.test.utils import db_available
from bob.bio.base.test.test_database_implementations import check_database


@db_available('replay')
def test_replay():
    module = bob.bio.base.load_resource('replay', 'config',
        preferred_package='bob.pad.face')
    try:
        check_database(module.database, protocol=module.protocol,
            groups=('train', 'dev', 'eval'))
    except IOError as e:
        raise SkipTest(
            "The database could not be queried; probably the db.sql3 file is missing. Here is the error: '%s'" % e)
