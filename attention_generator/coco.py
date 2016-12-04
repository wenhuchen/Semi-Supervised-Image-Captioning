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
    
    if options['with_glove']:
        y = numpy.zeros((len(feat_list), options['cutoff'], options['semantic_dim']), dtype='float32')
        for idx, ff in enumerate(feat_list):
            y[idx] = ff.reshape((-1, options['semantic_dim']))
    else:
        y = numpy.asarray(feat_list)
    
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
            f = h5py.File(osp.join(path, 'salient_feats_fc7_conv5_%s.h5'%opt), 'r')
            f_caps = read_json(osp.join(path, 'captions_%s.json'%opt))
            if options['semantic_dim'] == 2048:
                if 'regional_feats' in f.keys():
                    f_att = f['/regional_feats']
                else:
                    raise ValueError("The dataset doesn't contain regional features")
            elif options['semantic_dim'] == 300:
                if options['with_glove']:
                    f_att = f['/semantic_feats']
                else:
                    f_visual_concept = read_json(osp.join(path, 'visual_concept_%s.json'%opt))
                    f_att = numpy.zeros((len(f_visual_concept), options['cutoff']), dtype='int32')
                    for i, elem in enumerate(f_visual_concept):
                        f_att[i] = numpy.asarray([worddict[w] for w in elem['text'][:options['cutoff']]])
            else:
                raise ValueError("Unknown semantic dimension")
            if options['use_cnninit']:
                if options['cnn_type'] == 'vgg':
                    f_cnn = f['/fc7_feats']
                else:
                    f_cnn = h5py.File(osp.join(path, '%s_pool5_%s.h5'%(options['cnn_type'],opt)), 'r')['/pool5']
            else:
                f_cnn = numpy.zeros_like(f["/fc7_feats"][:], dtype='float32')
            exec "%s = (f_caps, f_att, f_cnn)"%opt
            data.append(opt)
    exec 'result =  (%s, worddict)'%(', '.join(data))
    return result

def generate_datasets(dataset_name, only_caps=True):
    batch_size = 64
    prefix = os.environ['WORK']
    basedir = '../Data/coco'
    caption_ref = read_json(osp.join(basedir, 'caption_ref.json'))
    vocab = read_pkl(osp.join(basedir, 'coco_dictionary.pkl'))
    salient_mapping = read_pkl('../Data/GloVe/vocab_pre_glove.pkl')
    file_mapping = read_pkl(osp.join(basedir, 'id2path_mapping.pkl'))
    detected_words_file = [osp.join(basedir, 'visual_concept_%s.json'%set_name) for set_name in dataset_name]
    vgg_deploy_path = '../Models/VGG_ILSVRC_16_layers_deploy.prototxt'
    vgg_model_path  = '../Models/VGG_ILSVRC_16_layers.caffemodel'
    lenet_deploy_path = '../Models/bvlc_googlenet_deploy.prototxt'
    lenet_model_path = '../Models/bvlc_googlenet.caffemodel'

    # Generate caption mapping
    caption_mapping = {}
    for set_name in dataset_name:
        if 'official' not in set_name:
            inputs = caption_ref['%s_cap'%set_name] 
            for input in inputs:
                if input['id'] not in caption_mapping:
                    caption_mapping[input['id']] = [input['text']]
                else:
                    caption_mapping[input['id']].append(input['text'])
    
    if not only_caps:
        from anandlib.dl.caffe_cnn import CNN
        # Initialize CNN
        cnn = CNN(deploy=vgg_deploy_path,
                  model=vgg_model_path,
                  batch_size=batch_size,
                  width=224,
                  height=224)

    # Training Data
    for index, name in enumerate(dataset_name):
        detected_words = read_json(detected_words_file[index])
        file_name = osp.join(prefix, 'semantic-LSTM', 'coco', 'salient_feats_fc7_conv5_%s.h5'%name)
        
        # Generate image related features
        if not only_caps:
            if not osp.exists(file_name):
                f = h5py.File(file_name, "w")
                semantic_flatten = f.create_dataset("semantic_feats", (len(detected_words), 30 * 300), dtype="float32")
                fc7_flatten = f.create_dataset("fc7_feats", (len(detected_words), 4096), dtype="float32")
                conv5_flatten = f.create_dataset("conv5_feats", (len(detected_words), 512, 14, 14), dtype='float32')
            else:
                f = h5py.File(file_name, "r+")
                semantic_flatten = f["/semantic_feats"]
                fc7_flatten = f["/fc7_feats"]
                conv5_flatten = f["/conv5_feats"]
            start_time = time.time()
            for start in range(0, len(detected_words), batch_size):
                end = min(start + batch_size, len(detected_words))
                img_path = [osp.join(prefix, 'coco-dataset', file_mapping[x['id']]) for x in detected_words[start:end]]
                fc7_features = cnn.get_features(image_list=img_path, layers='fc7',layer_sizes=[4096])
                conv5_features = cnn.get_features(image_list=img_path, layers='conv5_3',layer_sizes=[512, 14, 14])
                fc7_flatten[start:end, ...] = fc7_features[:end-start]
                conv5_flatten[start:end, ...] = conv5_features[:end-start]
                #if start % 5000 < start and start > 0:
                print "finished %d images "%start, " used time: %d"%(time.time()-start_time)
                start_time = time.time()
        
        # Generate captions and imageid
        caption_flatten = []
        start_time = time.time()
        for i, item in enumerate(detected_words):
            words, imgid = item['text'], item['id']
            if not only_caps:
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
        json.dump(caption_flatten, open(osp.join(basedir, 'captions_%s.json'%name), 'w'))

        if not only_caps:
            f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default='datasets')
    parser.add_argument('--only_caps', action='store_true', help="whether to generate image features")
    args = parser.parse_args()
    generate_datasets(args.datasets, args.only_caps)
