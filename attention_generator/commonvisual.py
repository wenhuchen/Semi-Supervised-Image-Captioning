import json
import sys
import pickle
import numpy
from random import shuffle
import h5py

index = 0
fname = sys.argv[1]
regional_mapping = pickle.load(open(sys.argv[2]))
h5_f = h5py.File(sys.argv[3], 'w')
f = open(fname, 'r').readlines()
#regional_feats = numpy.empty((0, 10*2048), 'float32')
h5_f.create_dataset('fc7_feats', dtype='float32', data=numpy.zeros((1, 4096)))
regional_feats = h5_f.create_dataset('regional_feats', dtype='float32', shape=(len(f), 10*2048))
# generate regional features
for i, line in enumerate(f):
    words = line.split()
    shuffle(words)
    result = numpy.asarray([regional_mapping[word] for word in words if word in regional_mapping], 'float32').flatten()
    tmp = numpy.zeros((10*2048,), 'float32')
    if result.shape[0] > 10*2048:
        tmp = result[:10*2048]
    else:
        tmp[:result.shape[0]] = result
    regional_feats[i] = tmp
    index += 1
    if index % 1000 == 0:
        print "finished %d captions"%index

h5_f.close()
