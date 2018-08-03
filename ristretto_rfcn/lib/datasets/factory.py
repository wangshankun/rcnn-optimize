# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""


#from datasets.pascal_voc import pascal_voc
#from datasets.coco import coco

#from datasets.inria import inria
from datasets.container_types import container_types
from datasets.container_letters import container_letters
from datasets.container_digits import container_digits
from datasets.roi_regions import roi_regions
from datasets.kitti import kitti 

from datasets.GasStation.gs_plate_roi import gs_plate_roi
from datasets.GasStation.gs_plate_char import gs_plate_char
from datasets.GasStation.gs_car_roi import gs_car_roi


import numpy as np

__sets = {}



#==========================GasStation============================

for split in ['train', 'test']:
    name = 'gs_plate_roi_{}'.format(split)
    __sets[name] = (lambda split=split: gs_plate_roi(split))

for split in ['train', 'test']:
    name = 'gs_car_roi_{}'.format(split)
    __sets[name] = (lambda split=split: gs_car_roi(split))

for split in ['train', 'test']:
    name = 'gs_plate_char_{}'.format(split)
    __sets[name] = (lambda split=split: gs_plate_char(split))

#==========================GasStation============================
#train container types
types_devkit_path = 'data/containertypes'
for split in ['train', 'test']:
    name = '{}_{}'.format('types', split)
    __sets[name] = (lambda split=split: container_types(split, types_devkit_path))

#train container letters
letters_devkit_path = 'data/containerletters'
for split in ['train', 'test']:
    name = '{}_{}'.format('letters', split)
    __sets[name] = (lambda split=split: container_letters(split, letters_devkit_path))

#train container digits
digits_devkit_path = 'data/containerdigits'
for split in ['train', 'test']:
    name = '{}_{}'.format('digits', split)
    __sets[name] = (lambda split=split: container_digits(split, digits_devkit_path))

#kitti
kitti_devkit_path = 'data/kitti'
for split in ['trainval', 'test']:
    name = '{}_{}'.format('kitti', split)
    __sets[name] = (lambda split=split: kitti(split, kitti_devkit_path))

#train container digits ROI regions
roi_devkit_path = 'data/ROIRegions'
for split in ['train', 'test']:
    name = '{}_{}'.format('roi', split)
    __sets[name] = (lambda split=split: roi_regions(split, roi_devkit_path))

#      # Set up voc_<year>_<split> using selective search "fast" mode
#      for year in ['2007', '2012']:
#          for split in ['train', 'val', 'trainval', 'test']:
#              name = 'voc_{}_{}'.format(year, split)
#              __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
#      
#      # Set up coco_2014_<split>
#      for year in ['2014']:
#          for split in ['train', 'val', 'minival', 'valminusminival']:
#              name = 'coco_{}_{}'.format(year, split)
#              __sets[name] = (lambda split=split, year=year: coco(split, year))
#      
#      # Set up coco_2015_<split>
#      for year in ['2015']:
#          for split in ['test', 'test-dev']:
#              name = 'coco_{}_{}'.format(year, split)
#              __sets[name] = (lambda split=split, year=year: coco(split, year))
#      
#      inria_devkit_path = '/home/westwell/Documents/py-faster-rcnn/data/INRIA_Person_devkit'
#      for split in ['train', 'test']:
#          name = '{}_{}'.format('inria', split)
#          __sets[name] = (lambda split=split: inria(split, inria_devkit_path))
#      

def get_imdb(name):
    """Get an imdb (image database) by name."""
    # __sets['PersonTrain'] = (lambda imageset = imageset, devkit = devkit: datasets.inria(imageset,devkit))
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
