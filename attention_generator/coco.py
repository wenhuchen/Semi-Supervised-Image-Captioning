import cPickle as pkl
import os
import sys
import os.path as osp
import h5py
import numpy
import json
import _init_paths
import argparse
from util import read_json, read_pkl
import time

def prepare_data(caps, features, cnn_features, worddict, options, zero_pad=False):
    # Caps is the captions regarding to ith image. Features is the batch x salient_words x emb_size vector
    seqs = []
    feat_list = []
    cnn_feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if w in worddict else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]][:options['cutoff']*options['semantic_dim']])
        if cnn_features.shape[0] == 1:
            cnn_feat_list.append(cnn_features[0])
        else:
            cnn_feat_list.append(cnn_features[cc[1]])

    lengths = [len(s) for s in seqs]
    
    if options['maxlen'] != None:
        new_seqs,new_feat_list,new_cnn_feat_list,new_lengths = [], [], [], []
        for l, s, y, c in zip(lengths, seqs, feat_list, cnn_feat_list):
            if l <= options['maxlen']:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_cnn_feat_list.append(c)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        cnn_feat_list = new_cnn_feat_list
        seqs = new_seqs
    
    y = numpy.zeros((len(feat_list), options['cutoff'], options['semantic_dim']), dtype='float32')
    for idx, ff in enumerate(feat_list):
        y[idx] = ff.reshape((-1, options['semantic_dim'])) 

    # CNN features
    y_cnn = numpy.array(cnn_feat_list)

    # Padding with zeros
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2]), dtype='float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.
    
    return x, x_mask, y, y_cnn

def load_data(path, options, load_train=False, load_val=False, load_test=False, load_official_val=False, load_official_test=False):
    print "Loading data"
    data = []
    worddict = read_pkl(osp.join(path, 'dictionary.pkl'))
    for opt in ('train', 'val', 'test', 'official_val', 'official_test'):
        if eval('load_%s'%opt):
            f = h5py.File(osp.join(path, 'Feats_%s.h5'%opt), 'r')
            f_caps = read_json(osp.join(path, 'captions_%s.json'%opt))
            if options['semantic_dim'] == 2048:
                if 'regional_feats' in f.keys():
                    f_att = f['/regional_feats']
                else:
                    raise ValueError("The dataset doesn't contain regional features")
            elif options['semantic_dim'] == 300:
                f_att = f['/semantic_feats']
            else:
                raise ValueError("Unknown semantic dimension")
            if options['use_cnninit']:
                if options['cnn_type'] == 'vgg':
                    f_cnn = f['/fc7_feats']
                else:
                    f_cnn = f['/pool5_feats']
            else:
                f_cnn = numpy.zeros_like(f["/fc7_feats"][:], dtype='float32')
            exec "%s = (f_caps, f_att, f_cnn)"%opt
            data.append(opt)
    exec 'result =  (%s, worddict)'%(', '.join(data))
    return result
