"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add coco-caption
coco_caption_folder = osp.join('../', 'coco-caption')
add_path(coco_caption_folder)

# Add config
config_folder = '../'
add_path(config_folder)

from config import *
print caffe_standard
add_path(caffe_standard)
import caffe
