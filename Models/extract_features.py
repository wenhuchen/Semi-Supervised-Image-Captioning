import _init_paths
import caffe
import cv2
import numpy as np
import skimage.transform
import pickle
import json
import os.path as osp
import lasagne
import argparse
import h5py
import time
from lasagne.utils import floatX
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer # can be replaced with dnn layers
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax

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
    return floatX(im[np.newaxis])

# Build simple block
def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad, 
                       use_bias=False, nonlin=rectify):
    net = []
    net.append((
            names[0], 
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride, 
                      flip_filters=False, nonlinearity=None) if use_bias 
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None, 
                           flip_filters=False, nonlinearity=None)
        ))
    
    net.append((
            names[1], 
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2], 
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))
    
    return dict(net), net[-1][0]

# Build residual blocks
def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False, 
                         upscale_factor=4, ix=''):
    net = {}
    
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']
    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)
    
    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)
    
    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)
    
    right_tail = net[last_layer_name]
    left_tail = incoming_layer
    
    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]
        
    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)
    
    return net, 'res%s_relu' % ix

# Build lasagne net
def lasagne_net(net_caffe):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    sub_net, parent_layer_name = build_simple_block( net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
                                                    64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    # Init the block size of net
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)
    block_size = ['a'] + ['b'+str(i+1) for i in range(7)]
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)
    block_size = ['a'] + ['b'+str(i+1) for i in range(35)]
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    # Build classifing layer
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0, 
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)
    print 'Total number of layers:', len(lasagne.layers.get_all_layers(net['prob']))
    # Transfer weight from caffe to lasagne
    layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))
    print 'Number of layers: %i' % len(layers_caffe.keys())
    for name, layer in net.items():    
        if name not in layers_caffe:
            print name, type(layer).__name__
            continue
        if isinstance(layer, BatchNormLayer):
            layer_bn_caffe = layers_caffe[name]
            layer_scale_caffe = layers_caffe['scale' + name[2:]]
            layer.gamma.set_value(layer_scale_caffe.blobs[0].data)
            layer.beta.set_value(layer_scale_caffe.blobs[1].data)
            layer.mean.set_value(layer_bn_caffe.blobs[0].data)
            layer.inv_std.set_value(1/np.sqrt(layer_bn_caffe.blobs[1].data) + 1e-4)
            continue
        if isinstance(layer, DenseLayer):
            layer.W.set_value(layers_caffe[name].blobs[0].data.T)
            layer.b.set_value(layers_caffe[name].blobs[1].data)
            continue
        if len(layers_caffe[name].blobs) > 0:
            layer.W.set_value(layers_caffe[name].blobs[0].data)
        if len(layers_caffe[name].blobs) > 1:
            layer.b.set_value(layers_caffe[name].blobs[1].data)
    return layer

if __name__ == '__main__':
    parser = argparse.ArgumentParser("caffe model generate features")
    parser.add_argument('--split', type=str, required=True, help='choose a split')
    parser.add_argument('--concepts', type=str, required=True, help='choose a concept file')
    parser.add_argument('--type', type=str, required=True, help='choose a mapping function')
    parser.add_argument('--dataset', type=str, required=True, help='choose a dataset')
    parser.add_argument('--index', type=int, default=0, help='choose a mapping function')
    parser.add_argument('--max', type=int, default=1, help='choose a mapping function')
    parser.add_argument('--mapping', type=str, help='choose a mapping function')
    args = parser.parse_args()
    
    if args.type == 'resnet':
        net_caffe = caffe.Net('./ResNet-152-deploy.prototxt', './ResNet-152-model.caffemodel', caffe.TEST)
        mean_values = np.load('mean_value.npz')['mean']
        feature_size = 2048
        name = 'pool5'
    elif args.type == 'googlenet':
        net_caffe = caffe.Net('./bvlc_googlenet_deploy.prototxt', './bvlc_googlenet.caffemodel', caffe.TEST)
        mean_values = np.asarray([[[104]], [[117]], [[123]]], dtype='float32')
        feature_size = 1024
        name = 'pool5/7x7_s1'
    
    f_visual_concept = json.load(open(args.concepts))
    if args.dataset == 'coco':
        mapping = pickle.load(open(args.mapping))
        prefix = '/work/wc396622/coco-dataset'
    elif args.dataset:
        mapping = None
        prefix = '/work/wc396622/flick30k-dataset/flickr30k-images'
    fname = '../Data/%s/%s_pool5_%s.h5'%(args.dataset, args.type, args.split)
    batch_size = 20
    if not osp.exists(fname):
        f = h5py.File(fname, 'w')
        f_pool5 = f.create_dataset('pool5', dtype='float32', shape=(len(f_visual_concept), feature_size))
    else:
        f = h5py.File(fname, 'r+')
        f_pool5 = f['/pool5']
    visual_concept_num = len(f_visual_concept)/args.max
    print "Generate captions for %d images"%visual_concept_num

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
