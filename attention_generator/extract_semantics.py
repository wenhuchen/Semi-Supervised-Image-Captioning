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

def generate_datasets(dataset, split, order):
    basedir = '../Data/%s'%dataset
    caption_ref = read_json(osp.join(basedir, 'caption_ref.json'))
    vocab = read_pkl(osp.join(basedir, 'dictionary.pkl'))
    salient_mapping = read_pkl('../Data/GloVe/vocab_pre_glove.pkl')
    file_mapping = read_pkl(osp.join(basedir, 'id2path_mapping.pkl'))
    detected_words_file = [osp.join(basedir, 'visual_concept_%s.json'%set_name) for set_name in split]

    # Generate caption mapping
    caption_mapping = {}
    for set_name in split:
        if 'official' not in set_name:
            inputs = caption_ref['%s_cap'%set_name] 
            for input in inputs:
                if input['id'] not in caption_mapping:
                    caption_mapping[input['id']] = [input['text']]
                else:
                    caption_mapping[input['id']].append(input['text'])
    
    # Training Data
    for index, name in enumerate(split):
        detected_words = read_json(detected_words_file[index])
        file_name = osp.join(basedir, 'feats_%s.h5'%name)
        
        # Generate image related features
        if not osp.exists(file_name):
            f = h5py.File(file_name, "w")
            semantic_flatten = f.create_dataset("semantic_feats", (len(detected_words), order * 300), dtype="float32")
        else:
            f = h5py.File(file_name, "r+")
            semantic_flatten = f["/semantic_feats"]

        # Generate captions and imageid
        caption_flatten = []
        start_time = time.time()
        
        # Generate semantic features and captions files
        for i, item in enumerate(detected_words):
            words, imgid = item['text'], item['id']
            semantic_flatten[i, ...] = numpy.hstack([salient_mapping[vocab[w]] for w in words])
            if len(caption_mapping) > 0:
                for j in range(5):
                    caption_flatten.append([caption_mapping[imgid][j], i, imgid])
            else:
                caption_flatten.append(["UNKNOWN SENTENCES", i, imgid])
            if i % 5000 == 0 and i > 0:
                print "used time: ", time.time() - start_time
                start_time = time.time()
                print "finished %d images"%i
        
        # Write into captions
        with open(osp.join(basedir, 'captions_%s.json'%name), 'w') as f:
            json.dump(caption_flatten, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, default='datasets')
    parser.add_argument('--split', type=str, nargs='+', default='datasets')
    args = parser.parse_args()
    generate_datasets(args.datasets, args.split, order)
