from __future__ import division
from _init_paths import *
import os
import os.path as osp
import sg_utils as utils
import numpy as np
import skimage.io
import skimage.transform
import h5py
import pickle
import json
import math
import argparse
import time
import cv2
from collections import Counter
from json import encoder
"""
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
"""
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
mean = np.array([[[ 103.939, 116.779, 123.68]]])
functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are', 'it'] 

def scaleandtranspose(im, base_image_size):
    # Subtract the ilvsr mean value
    new_im = im - mean
    # Upsample the image and swap the axes to Color x height x width
    new_im = upsample_image(new_im, base_image_size, square=True)
    return new_im.transpose((2,0,1))

def BGR2RGB(img):
    assert img.shape[2] == 3
    new_img = img.copy()
    new_img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    return new_img

def clip(lo, x, hi):
    return lo if x <= lo else hi if x >= hi else x

def data_crop(im, boxes):
    # Make sure the larger edge is 720 in length
    H, W = im.shape[0], im.shape[1]
    bbox_img = im.copy()
    crop_list = []
    for box in boxes:
        # Careful that the order is height precede width
        leftup_x = clip(0, box[0], W)
        leftup_y = clip(0, box[1], H)
        rightbot_x = clip(0, box[0] + box[2], W)
        rightbot_y = clip(0, box[1] + box[3], H)
        crop_list.append(im[leftup_y:rightbot_y, leftup_x:rightbot_x, :])
        cv2.rectangle(bbox_img, (leftup_x, leftup_y), (rightbot_x, rightbot_y), (0, 255, 0), 2)
    return crop_list, bbox_img

def upsample_image(im, upsample_size, square=False):
    h, w = im.shape[0], im.shape[1]
    s = max(h, w)
    if square:        
        I_out = np.zeros((upsample_size, upsample_size, 3), dtype=np.float) 
    else:
        new_h = math.ceil(h/w * upsample_size) if w>=h else upsample_size
        new_w = math.ceil(w/h * upsample_size) if h>=w else upsample_size 
        I_out = np.zeros((new_h, new_w, 3), dtype=np.float)
    im = cv2.resize(im, None, None, fx = upsample_size/s, fy = upsample_size/s, interpolation=cv2.INTER_CUBIC)
    I_out[:im.shape[0], :im.shape[1], :] = im
    return I_out 

def filter_out(concepts):
    rank = Counter() 
    for concept in concepts:
        rank.update(concept)
    words = map(lambda arg: arg[0], rank.most_common(20))
    return words

class DataLoader(object):
    def __init__(self, coco_h5, coco_json):
        self.h5 = h5py.File(coco_h5)
        self.label_start_ix = self.h5['label_start_ix']
        self.label_end_ix = self.h5['label_end_ix']
        self.json_image = json.load(open(coco_json))['images']
        self.image_num = len(json.load(open(coco_json))['images'])
        self.ix_to_word = json.load(open(coco_json))['ix_to_word']
        self.split_ix = {}
        self.seq_length = 16
        self.iterator = {}
        for i, info in enumerate(self.json_image):
            if info['split'] not in self.split_ix:
                self.split_ix[info['split']] = [i]
            else:
                self.split_ix[info['split']].append(i)
        self.reset_iterator()
    
    def get_image_num(self, split):
        if split == 'train':
            return self.image_num - 10000
        else:
            return 5000

    def reset_iterator(self):
        for k in self.split_ix.keys():
            self.iterator[k] = 0

    def get_batch(self, split, batch_size=1, seq_per_img=5, seq_length=16):
        images = np.zeros((batch_size, 256, 256, 3))
        seq = np.zeros((seq_per_img, seq_length))
        split_ix = self.split_ix[split]
        max_ix = self.h5['labels'].shape[0]
        max_index = len(split_ix)
        wrapped = False
        info = []
        for i in range(batch_size):
            ri = self.iterator[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            self.iterator[split] = ri_next
            ix = split_ix[ri]

            ix1 = self.h5['label_start_ix'][ix]
            ix2 = self.h5['label_end_ix'][ix]
            ncaps = ix2 - ix1 + 1
            assert ncaps > 0
            if ncaps >= seq_per_img:
                rand_ix = np.random.choice(range(ix1, ix2+1), seq_per_img, replace=False)
            else:
                rand_ix = np.random.choice(range(ix1, ix2+1), seq_per_img, replace=True)
            for j, j_ix in enumerate(rand_ix):
                if j_ix >= max_ix:
                    seq[j] = self.h5['labels'][-1, :seq_length]
                else:
                    seq[j] = self.h5['labels'][j_ix, :seq_length]
            im = self.h5['images'][ix].astype(np.float32)
            images[i] = np.transpose(im, axes = (1, 2, 0))
            info.append({'id': self.json_image[ix]['id'], 'file_path': self.json_image[ix]['file_path']})
        return images, seq, info, wrapped

class TestModel(object):
    def __init__(self, vocab_file):
        # Set threshold_metric_name and output_metric_name
        self.base_image_size = 565
        self.vocab = utils.load_variables(vocab_file)
        self.is_functional = np.array([x not in functional_words for x in self.vocab['words']])
        self.threshold = 0.5

    def load(self, prototxt_deploy, model_file):
        self.net = caffe.Net(prototxt_deploy, model_file, caffe.TEST)
    
    def forward(self, im, order):
        # Make sure the image passed in are BGR order and height x width x channel order
        self.net.forward(data=im)

        # Retrieve the mil probability of the word
        mil_probs = self.net.blobs['mil'].data
        mil_probs = mil_probs.reshape((mil_probs.shape[0], mil_probs.shape[1]))
        top_ind = np.argsort(-mil_probs, axis=-1)[:, :order + len(functional_words)]

        # If not for regional features, just return the distribution
        if order == 1000:
            return self.net.blobs['mil'].data
        
        # Retrive the sigmoid data from the sigmoid layer
        fc8_conv_probs = self.net.blobs['fc8-conv-sigmoid'].data
        fc8_conv = fc8_conv_probs.reshape((fc8_conv_probs.shape[0], fc8_conv_probs.shape[1], -1))
        fc8_conv_arg = fc8_conv.argmax(axis=-1)

        # Retrive the correponding feature maps
        feat_map = self.net.blobs['fc7-conv'].data
        concepts, prob  = [], []
        att_feat = np.zeros((feat_map.shape[0], order, feat_map.shape[1]), dtype='float32') 
        feat_probs = np.zeros((feat_map.shape[0], order, 12, 12), dtype='float32')

        # Loop over all the sorted indexes
        indexes = []
        for i in range(top_ind.shape[0]):
            tmp_concepts  = []
            for j in range(top_ind.shape[1]):
                word_idx = top_ind[i, j]
                prob_map = fc8_conv_probs[i, word_idx, :, :]
                index = fc8_conv_arg[i, word_idx]
                word = self.vocab['words'][word_idx]
                if word not in functional_words:
                    if index not in indexes:
                        i1, i2 = divmod(index, 12)
                        att_feat[i, len(indexes)] = feat_map[i,:,i1,i2]
                        indexes.append(index)
                    feat_probs[i, len(tmp_concepts)] = prob_map
                    tmp_concepts.append(word)
                if len(tmp_concepts) >= order:
                    break
            concepts.append(tmp_concepts)
            prob.append(mil_probs[i, top_ind[i]].tolist())

        return concepts, prob, att_feat, feat_probs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("run visual concept extraction")
    parser.add_argument("--test_json", type=str, required=True, help="test image json")
    parser.add_argument("--dataset", type=str, required=True, help="the dataset to use")
    parser.add_argument("--split", type=str, required=True, help="Choose a split to evaluate")
    parser.add_argument("--order", type=int, default=20, help="test image json")
    parser.add_argument("--gpuid", type=int, default=0, help="GPU id to run")
    parser.add_argument("--salient_grt", type=str, default='../Data/coco/salient_grt.json', help="Groundtruth")
    parser.add_argument("--batch_size", type=int, default=1, help="Verbose the results")
    parser.add_argument("--verbose", action='store_true', help="Verbose the results")
    args = parser.parse_args()
    
    # Caffe setting
    caffe.set_mode_gpu()
    caffe.set_device(args.gpuid)
    prototxt = 'output/vgg/mil_finetune.prototxt.deploy'
    model_file = 'output/vgg/snapshot_iter_240000.caffemodel'
    vocab_file = 'vocabs/vocab_train.pkl'
    basedir = '../Data/%s'%args.dataset
    prefix = coco_image_base if dataset=='coco' else flickr_image_base
    #prototxt = '/home/thes0193/code/output/v2/mil_finetune.prototxt.deploy'
    #model_file = '/home/thes0193/code/output/v2/snapshot_iter_240000.caffemodel'
    #vocab_file = '/home/thes0193/code/vocabs/new_train_vocab.pkl'

    # Load the model
    model = TestModel(vocab_file)
    with open(args.salient_grt) as f:
        salient_grt_map = {item['id']:item['words'] for item in json.load(f)} 
    model.load(prototxt, model_file)

    # Open h5 file, if not exist then create one, if exists just load it
    image_f = json.load(open(args.test_json))
    result, prec_set = [], []
    h5_name = osp.join(basedir, 'Feats_%s.h5'%(args.split))
    if osp.exists(h5_name):
        h5_f = h5py.File(h5_name, 'r+')
    else:
        h5_f = h5py.File(h5_name, 'w')
    if 'regional_feats' not in h5_f.keys():
        feats = h5_f.create_dataset('regional_feats', shape=(len(image_f), args.order*2048), dtype='float32')
    else:
        feats = h5_f['/regional_feats']

    # Start generate results, i.e. visual concepts and regionl features
    for start in range(0, len(image_f), args.batch_size):
        end = min(start+args.batch_size, len(image_f))
        img_batch = np.zeros((args.batch_size, 3, 565, 565), 'float32')
        for i in range(start, end):
            img = cv2.imread(osp.join(prefix, image_f[i]['file_name']))
            img_batch[i-start] = scaleandtranspose(img, 565)
        results =  model.forward(img_batch, args.order)
        # Calculate the precision and recall
        for i in range(start, end):
            # Calculate precision
            if image_f[i]['id'] in salient_grt_map:
                ref = salient_grt_map[image_f[i]['id']]
                prec = len(set(ref) & set(results[0][i-start]))/len(ref)
                prec_set.append(prec)
                print "Precision: %0.2f"%(sum(prec_set)/len(prec_set))
            # Form results
            result.append({'id': f[i]['id'], 'text': results[0][i-start], 'prob': results[1][i-start]})
            feats[start:end] = results[2][:,:,::2].reshape((args.batch_size, -1))
            """
            img_fig = plt.figure()
            plt.imshow(BGR2RGB(origin_img[i]))
            plt.axis('off')
            fig = plt.figure(figsize=(10, 6), facecolor='white')
            for j in range(12):
                img = (batch_img[i].transpose((1,2,0)) + mean)/255
                ax = fig.add_subplot(3, 4, j+1)
                #ax.set_axis_bgcolor('white')
                ax.imshow(BGR2RGB(img))
                alpha_img = skimage.transform.resize(feat_probs[i,j], [img.shape[0], img.shape[1]])
                ax.imshow(alpha_img, cmap=cm.Greys_r, alpha=0.8)
                ax.set_title(visual_concepts[i][j])
                ax.axis('off')
            plt.show()
            raw_input("Press Enter to continue...")
            """
            if start % 100 == 0 and start > 0: 
                print "Finished %d images"%start
    h5_f.close()
    # Dump it into the visual concept files for next step
    with open(osp.join(basedir,'visual_concept_%s.json'%args.split), 'w') as f:
        pickle.dump(result, f)
