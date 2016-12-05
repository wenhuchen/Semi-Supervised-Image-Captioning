from _init_paths import *
import cv2
import numpy as np
import pickle
import json
import os.path as osp
import skimage.transform
import argparse
import h5py
import time

# Preprocess image
def prep_image(fname, mean_values):
    im = cv2.imread(fname)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    im = im - mean_values
    return im[np.newaxis].astype('float32')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("caffe model generate features")
    parser.add_argument('--split', type=str, required=True, help='choose a split')
    parser.add_argument('--concepts', type=str, required=True, help='choose a concept file')
    parser.add_argument('--type', type=str, required=True, help='choose a cnn type')
    parser.add_argument('--dataset', type=str, required=True, help='choose a dataset')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--max', type=int, default=1)
    parser.add_argument('--mapping', type=str, help='choose a mapping function')
    args = parser.parse_args()
    
    # Specify the caffe file name and batch size
    if args.type == 'resnet':
        net_caffe = caffe.Net('../Models/ResNet-152-deploy.prototxt', '../Models/ResNet-152-model.caffemodel', caffe.TEST)
        mean_values = np.load('mean_value.npz')['mean']
        feature_size = 2048
        name = 'pool5_feats'
        batch_size = 20
    elif args.type == 'vgg':
        net_caffe = caffe.Net('../Models/vgg-16-deploy.prototxt', '../Models/vgg-16-model.caffemodel', caffe.TEST)
        mean_values = np.load('mean_value.npz')['mean']
        feature_size =4096
        name = 'fc7_feats'
        batch_size = 50
    
    # Load f_visual_concept, used to extract image id
    f_visual_concept = json.load(open(args.concepts))
    if args.dataset == 'coco':
        mapping = pickle.load(open(args.mapping))
        prefix = coco_image_base  # imported from config.py
    elif args.dataset == 'flickr':
        mapping = None
        prefix = flickr_image_base # imported from config.py 
    
    # Specify the h5 file, noramlly it should already exist, we need to add dataset into it
    fname = '../Data/%s/feats_%s.h5'%(args.dataset, args.split)
    if not osp.exists(fname):
        f = h5py.File(fname, 'w')
    else:
        f = h5py.File(fname, 'r+')
    if name in f.keys():
        cnn_dataset = f['/%s'%name]
    else:
        cnn_dataset = f.create_dataset(name, dtype='float32', shape=(len(f_visual_concept), feature_size))
    
    # Retrieve the number of images
    visual_concept_num = len(f_visual_concept)/args.max
    print "Generate captions for %d images"%visual_concept_num

    # Start generating iamges
    tik = time.time()
    for start in range(args.index*visual_concept_num, (args.index+1)*visual_concept_num, batch_size):
        end = min(start + batch_size, (args.index+1)*visual_concept_num)
        im = np.zeros((batch_size, 3, 224, 224), dtype='float32')
        for i in range(start, end):
            path = '%d.jpg'%f_visual_concept[i]['id'] if mapping is None else mapping[f_visual_concept[i]['id']]
            im[i-start] = prep_image(osp.join(prefix, path), mean_values)
        net_caffe.forward(data=im)
        f_pool5[start:end] = net_caffe.blobs[name].data.squeeze()[:end-start]
        print "finished %d/%d within time %d"%(start-args.index*visual_concept_num, visual_concept_num, time.time() - tik)
        tik = time.time()
    f.close()
