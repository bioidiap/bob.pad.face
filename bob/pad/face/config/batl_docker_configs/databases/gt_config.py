#!/usr/bin/env python

ORIGINAL_DIRECTORY = "/tmp/sub_dir/data/"
ANNOTATIONS_TEMP_DIR = "/tmp/sub_dir/output/train/annotations/"
N_FRAMES = 50

GT_PATH = '/tmp/sub_dir/gt.csv'
IDIAP_DATA_GT_PATH = '/tmp/idiap_extracted_data/gt_idiap.csv'

GT_CONFIG = dict(path=0, any_pa=1, face=2)
IDIAP_GT_CONFIG = dict(path=0, type_id=1, pai_id=2, low_level_group=4)

GROUND_TRUTH = {'govt' : {'path' : GT_PATH,
                          'config' : GT_CONFIG
                         },
                'idiap': {'path' : IDIAP_DATA_GT_PATH,
                          'config': IDIAP_GT_CONFIG
                         }
               }
