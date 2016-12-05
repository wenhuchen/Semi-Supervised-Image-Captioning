"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add config
config_folder = '../'
add_path(config_folder)

from config import *
print caffe_mil
add_path(caffe_mil)
import caffe
