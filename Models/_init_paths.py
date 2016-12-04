"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add caffe to PYTHONPATH
caffe_path = osp.join(os.environ['HOME'], "py-faster-rcnn/caffe-fast-rcnn", "python")
add_path(caffe_path)
