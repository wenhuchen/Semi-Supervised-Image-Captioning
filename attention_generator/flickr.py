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
from coco import prepare_data, load_data

def generate_datasets(dataset_name, resume=0):
    batch_size = 8
    prefix = os.environ['WORK']
    basedir = '../Data/flickr'
    caption_ref = read_json(osp.join(basedir, 'caption_ref.json'))
    vocab = read_pkl(osp.join(basedir, 'dictionary.pkl'))
    salient_mapping = read_pkl('../Data/GloVe/vocab_glove.pkl')
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
        file_name = osp.join(prefix, 'semantic-LSTM', 'flickr', 'salient_feats_fc7_conv5_%s.h5'%name)
        
        # Generate image related features
        f = h5py.File(file_name, "w")
        semantic_flatten = f.create_dataset("semantic_feats", (len(detected_words), 20 * 300), dtype="float32")
        fc7_flatten = f.create_dataset("fc7_feats", (len(detected_words), 4096), dtype="float32")
        #conv5_flatten = f.create_dataset("conv5_feats", (len(detected_words), 512, 14, 14), dtype='float32')
        start_time = time.time()
        for start in range(resume, len(detected_words), batch_size):
            end = min(start + batch_size, len(detected_words))
            img_path = [osp.join(prefix, 'flick30k-dataset', 'flickr30k-images', '%d.jpg'%x['id']) for x in detected_words[start:end]]
            fc7_features = cnn.get_features(image_list=img_path, layers='fc7',layer_sizes=[4096])
            fc7_flatten[start:end] = fc7_features[:end-start]
            #conv5_features = cnn.get_features(image_list=img_path, layers='conv5_3',layer_sizes=[512, 14, 14])
            #conv5_flatten[start:end, ...] = conv5_features[:end-start]
            #if start % 5000 < start and start > 0:
            print "finished %d images "%start, " used time: %d"%(time.time()-start_time)
            start_time = time.time()
        
        # Generate captions and imageid
        caption_flatten = []
        start_time = time.time()
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
        json.dump(caption_flatten, open(osp.join(basedir, 'captions_%s.json'%name), 'w'))

        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default='datasets')
    parser.add_argument('--resume', type=int, default=0, help='the resume point')
    args = parser.parse_args()
    generate_datasets(args.datasets, resume=args.resume)
