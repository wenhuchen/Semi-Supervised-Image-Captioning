import json
import cPickle as pkl
import os.path as osp
import numpy
import cPickle
import argparse
import h5py
from util import read_json, read_pkl
from coco import load_data, prepare_data

def process_text(line, worddict, length):
    word_idx = [worddict[w] if w in worddict else 1 for w in line.strip().split()] 
    word_len = len(word_idx)
    if word_len >= 15:
        word_idx = word_idx[:15]
    else:
        word_idx += [0]*(15-word_len)
    return word_idx

def generate_datasets(splits, doc_type, cutoff):
    captions_groundtruth = [] 
    basedir = '../Data/%s'%doc_type

    # Read the data
    for name in splits:
        with open(osp.join(basedir, 'commoncrawl.%s.txt'%name)) as f:
            captions_groundtruth.append(f.readlines())
    salient_mapping = read_pkl('../Data/GloVe/vocab_glove.pkl')
    worddict = read_pkl(osp.join(basedir,'dictionary.pkl'))
    
    # Fake the semantic features from the groundtruth
    for split, captions in zip(splits, captions_groundtruth):
        f = h5py.File(osp.join(basedir, "Feats_%s.h5"%split), "w")
        semantic_flatten = f.create_dataset("semantic_feats", (len(captions), cutoff * 300), dtype="float32")
        cnn_feats = f.create_dataset("fc7_feats", data=numpy.zeros((1, 4096), dtype='float32'))
        caption_flatten = []
        for i, line in enumerate(captions):
            if split == 'test':
                imgid = i + 5000
            elif split == 'train':
                imgid = i + 10000
            else:
                imgid = i
            word_idx = process_text(line, worddict, cutoff)
            semantic_flatten[i] = numpy.hstack([salient_mapping[w] for w in word_idx])
            caption_flatten.append([line.strip(), i, imgid])
            if i % 10000 == 0:
                print "Finished %d sentences"%i
        with open(osp.join(basedir, 'captions_%s.json'%split), 'w') as cap_f:
            json.dump(caption_flatten, cap_f)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=str, nargs='+', default='datasets')
    parser.add_argument('--type', type=str, required=True, help='type of commoncrawl small | large | noise')
    parser.add_argument('--cutoff', type=int, default=15, help='cut off concept number')
    args = parser.parse_args()
    generate_datasets(args.splits, args.type, args.cutoff)
