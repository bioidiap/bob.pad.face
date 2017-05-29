#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from bob.bio.base.grid import Grid

# Configuration to run on computation cluster:
idiap = Grid(
    training_queue='32G',

    number_of_preprocessing_jobs=32,
    preprocessing_queue='4G-io-big',

    number_of_extraction_jobs=32,
    extraction_queue='8G-io-big',

    number_of_projection_jobs=32,
    projection_queue='8G-io-big',

    number_of_enrollment_jobs=32,
    enrollment_queue='8G-io-big',

    number_of_scoring_jobs=1,
    scoring_queue='8G-io-big',
    )

# Configuration to run on user machines:
idiap_user_machines = Grid(
    training_queue='32G',

    number_of_preprocessing_jobs=32,
    preprocessing_queue='4G',

    number_of_extraction_jobs=32,
    extraction_queue='8G',

    number_of_projection_jobs=32,
    projection_queue='8G',

    number_of_enrollment_jobs=32,
    enrollment_queue='8G',

    number_of_scoring_jobs=1,
    scoring_queue='8G',
    )
