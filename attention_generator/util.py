import numpy
import copy
from collections import OrderedDict
import theano
import theano.tensor as tensor
import numpy as np
import json
import cPickle as pkl

##################################################
############## MATHEMATICAL HELPERS ##############
##################################################

def read_json(name):
    return json.load(open(name, 'r'))

def read_pkl(name):
    return pkl.load(open(name, 'r'))

def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

# some useful shorthands
def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

##################################################
############# THEANO MODEL HELPERS ###############
##################################################

'''
Theano uses shared variables for parameters, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy
dictionary
'''

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# get the average of a list 
def avlist(l):
    return reduce(lambda x, y: x + y, l) / len(l)

# dropout in theano
def dropout_layer(state_before, use_noise, trng):
    """
    tensor switch is like an if statement that checks the
    value of the theano shared variable (use_noise), before
    either dropping out the state_before tensor or
    computing the appropriate activation. During training/testing
    use_noise is toggled on and off.
    """
    proj = tensor.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)

# load parameters
def load_params(path, params, complete=True):
    print "Loading parameters from file: %s" % path
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp and complete:
            raise Warning('%s is not in the archive' % kk)
        elif kk not in pp and not complete:
            pass
        elif params[kk].shape == pp[kk].shape:
            params[kk] = pp[kk]
        elif not complete:
            pass
        else:
            raise Warning('Shape of %s is not correct' % kk)

    return params

def load_tparams(path, tparams):
    print "Loading parameters from file: %s" % path
    pp = numpy.load(path)
    for kk, vv in tparams.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        tparams[kk].set_value(pp[kk])

    return tparams

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


##################################################
############## PREPROCESSING CODE ################
##################################################

class HomogeneousData():
    def __init__(self, data, batch_size=128, maxlen=None):
        self.batch_size = 128
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0]

        # find the unique lengths
        self.lengths = [len(cc[0].split()) for cc in self.caps]
        self.len_unique = numpy.unique(self.lengths)
        # remove any overly long captions
        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

        # current counter
        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        # randomly choose the length
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

        caps = [self.caps[ii] for ii in curr_indices]

        return caps 

    def __iter__(self):
        return self

def preprocess(im):
    im = im.astype('float32')
    im[...,0] -= 103.939
    im[...,1] -= 116.779
    im[...,2] -= 123.68 
    im = im.transpose((0,3,1,2))
    return im

def seqs2words(cc, word_idict):
    ww = []
    for w in cc:
        if w == 0:
            break
        ww.append(word_idict[w])
    return ' '.join(ww)
