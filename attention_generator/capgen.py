'''
Source code for an attention based image caption generation system described
in:

Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
International Conference for Machine Learning (2015)
http://arxiv.org/abs/1502.03044

Comments in square brackets [] indicate references to the equations/
more detailed explanations in the above paper.
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy
import copy

from collections import OrderedDict
import warnings

from util import ortho_weight, norm_weight, tanh, rectifier, linear, avlist
from util import dropout_layer, _p
from layers import *

##################################################
################ PREPROCESSING ###################
##################################################
import coco
import commoncrawl
import flickr

# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = {'coco': (coco.load_data, coco.prepare_data),
            'commoncrawl': (commoncrawl.load_data, commoncrawl.prepare_data),
            'commoncrawl_addnoise': (commoncrawl.load_data, commoncrawl.prepare_data),
            'commoncrawl_larger': (commoncrawl.load_data, commoncrawl.prepare_data),
            'commonvisual': (commoncrawl.load_data, commoncrawl.prepare_data),
            'flickr': (flickr.load_data, flickr.prepare_data)}

def get_dataset(name):
    return datasets[name][0], datasets[name][1]


##################################################
############## NEURAL NETWORK DEF ################
##################################################


"""
Neural network layer definitions.

The life-cycle of each of these layers is as follows
    1) The param_init of the layer is called, which creates
    the weights of the network.
    2) The fprop is called which builds that part of the Theano graph
    using the weights created in step 1). This automatically links
    these variables to the graph.

Each prefix is used like a key and should be unique
to avoid naming conflicts when building the graph.
"""

#layers: 'name': ('parameter initializer', 'fprop')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          'lstm_cond_nox':('param_init_lstm_cond_nox', 'lstm_cond_nox_layer')
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

##################################################
################ INITIALIZATIONS #################
##################################################

# parameter initialization
# [roughly in the same order as presented in section 3.1.2]
# See above get_layer function + layers var for neural network definition

def init_params(options):
    params = OrderedDict()
    
    # Visual concept embedding
    if not options['with_glove']:
        params['VCemb'] = norm_weight(options['n_words'], options['dim_word'])
    # embedding: [matrix E in paper]
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    #params = get_layer('ff')[0](options, params, prefix='CNNTrans', nin=options['cnn_dim'], nout=options['dim'])
    ctx_dim = options['ctx_dim']
    
    if options['lstm_encoder']: # potential feature that runs an LSTM over the annotation vectors
        # use input attentive encoder
        params = get_layer('lstm_cond_nox')[0](options, params, prefix='encoder', dim=ctx_dim, dimctx=options['semantic_dim'])

    # potentially deep decoder (warning: should work but somewhat untested)
    for lidx in range(options['n_layers_lstm']):
        ff_state_prefix = 'CNNTrans_%d'%lidx if lidx > 0 else 'CNNTrans'
        ff_memory_prefix = 'CNN_memory_%d'%lidx if lidx > 0 else 'CNN_memory'
        lstm_prefix = 'decoder_%d'%lidx if lidx > 0 else 'decoder'
        nin_lstm = options['dim'] if lidx > 0 else options['dim_word']
        params = get_layer('ff')[0](options, params, prefix=ff_state_prefix, nin=options['cnn_dim'], nout=options['dim'])
        params = get_layer('ff')[0](options, params, prefix=ff_memory_prefix, nin=options['cnn_dim'], nout=options['dim'])
        params = get_layer('lstm_cond')[0](options, params, prefix=lstm_prefix,
                                           nin=nin_lstm, dim=options['dim'],
                                           dimctx=ctx_dim)

    # readout: [equation (7)]
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'])
    if options['ctx2out']:
        params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=ctx_dim, nout=options['dim_word'])
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            params = get_layer('ff')[0](options, params, prefix='ff_logit_h%d'%lidx, nin=options['dim_word'], nout=options['dim_word'])
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])

    return params

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params

##################################################
############### LAYER DEFINITIONS ################
##################################################

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

##################################################
# build a training model
def build_model(tparams, options):
    """ Builds the entire computational graph used for training

    Basically does a forward pass through the data and calculates the cost function

    [This function builds a model described in Section 3.1.2 onwards
    as the convolutional feature are precomputed, some extra features
    which were not used are also implemented here.]

    Parameters
    ----------
    tparams : OrderedDict
        maps names of variables to theano shared variables
    options : dict
        big dictionary with all the settings and hyperparameters
    Returns
    -------
    trng: theano random number generator
        Used for dropout, etc
    use_noise: theano shared variable
        flag that toggles noise on and off
    [x, mask, ctx, cnn_features]: theano variables
        Represent the captions, binary mask, and annotations
        for a single batch (see dimensions below)
    alphas: theano variables
        Attention weights
    alpha_sample: theano variable
        Sampled attention weights used in REINFORCE for stochastic
        attention: [see the learning rule in eq (12)]
    cost: theano variable
        negative log likelihood
    opt_outs: OrderedDict
        extra outputs required depending on configuration in options
    """
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples,
    x = tensor.matrix('x', dtype='int64')
    # mask: #samples,
    mask = tensor.matrix('mask', dtype='float32')
    # context: #samples x #visual_words x dim
    if options['with_glove']:
        ctx = tensor.tensor3('ctx', dtype='float32')
        new_ctx = ctx
    else:
        ctx = tensor.matrix('ctx', dtype='int32')
        new_ctx = tparams['VCemb'][ctx]
    # fc7 features: #samples x dim
    cnn_features = tensor.matrix('cnn_feats', dtype='float32')

    # index into the word embedding matrix, shift it forward in time, the first element is zero
    # Time step x S x D
    emb = tparams['Wemb'][x.flatten()].reshape([x.shape[0], x.shape[1], options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # forward-backward lstm encoder
    if options['lstm_encoder']:
        rval, encoder_alphas = get_layer('lstm_cond_nox')[1](tparams, options, prefix='encoder', context=new_ctx)
        ctx0 = rval.dimshuffle(1,0,2)
    else:
        ctx0 = new_ctx

    for lidx in range(options['n_layers_lstm']):
        init_state_prefix = 'CNNTrans_%d'%lidx if lidx > 0 else 'CNNTrans'
        init_memory_prefix = 'CNN_memory_%d'%lidx if lidx > 0 else 'CNN_memory'
        lstm_prefix = 'decoder_%d'%lidx if lidx > 0 else 'decoder'
        lstm_inps = proj_h if lidx > 0 else emb
        init_state = get_layer('ff')[1](tparams, cnn_features, options, prefix=init_state_prefix, activ='tanh')
        init_memory = get_layer('ff')[1](tparams, cnn_features, options, prefix=init_memory_prefix, activ='tanh')
        attn_updates = []
        proj, updates = get_layer('lstm_cond')[1](tparams, lstm_inps, options,
                                              prefix=lstm_prefix,
                                              mask=mask, context=ctx0,
                                              one_step=False,
                                              init_state=init_state,
                                              init_memory=init_memory,
                                              trng=trng,
                                              use_noise=use_noise)
        attn_updates += updates
        proj_h = proj[0]

    alphas = proj[2]
    ctxs = proj[4]

    if options['use_dropout']:
        proj_h = dropout_layer(proj_h, use_noise, trng)

    # compute word probabilities
    # [equation (7)]
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, trng)

    # compute softmax
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # Index into the computed probability to give the log likelihood
    x_flat = x.flatten()
    p_flat = probs.flatten()
    cost = -tensor.log(p_flat[tensor.arange(x_flat.shape[0])*probs.shape[1]+x_flat]+1e-8)
    cost = cost.reshape([x.shape[0], x.shape[1]])
    masked_cost = cost * mask
    #align_cost = (-standard_aligns*alphas).sum(2)

    cost = masked_cost.sum(0)
    
    # optional outputs
    opt_outs = dict()
    if options['lstm_encoder']:
        return trng, use_noise, [x, mask, ctx, cnn_features], [alphas, encoder_alphas], cost, opt_outs
    else:
        return trng, use_noise, [x, mask, ctx, cnn_features], [alphas], cost, opt_outs

# build a sampler
def build_sampler(tparams, options, use_noise, trng):
    """ Builds a sampler used for generating from the model
    Parameters
    ----------
    tparams : OrderedDict
        maps names of variables to theano shared variables
    options : dict
        big dictionary with all the settings and hyperparameters
    use_noise: boolean
        If true, add noise to the sampling
    trng: random number generator
    Returns
    -------
    f_init : theano function
        Input: annotation, Output: initial lstm state and memory
        (also performs transformation on ctx0 if using lstm_encoder)
    f_next: theano function
        Takes the previous word/state/memory + ctx0 and runs ne
        step through the lstm (used for beam search)
    """
    # context: #annotations x dim
    if options['with_glove']:
        ctx = tensor.matrix('ctx_sampler', dtype='float32')
        new_ctx = ctx
    else:
        ctx = tensor.vector('ctx_sampler', dtype='int32')
        new_ctx = tparams['VCemb'][ctx]
    if options['lstm_encoder']:
        ctx0, _ = get_layer('lstm_cond_nox')[1](tparams, options, prefix='encoder', context=new_ctx)
    else:
        ctx0 = new_ctx
    # initial state/cell
    cnn_features = tensor.vector('x_feats', dtype='float32')
    init_state, init_memory = [], []
    for lidx in range(options['n_layers_lstm']):
        init_state_prefix = 'CNNTrans_%d'%lidx if lidx > 0 else 'CNNTrans'
        init_memory_prefix = 'CNN_memory_%d'%lidx if lidx > 0 else 'CNN_memory'
        init_state.append(get_layer('ff')[1](tparams, cnn_features, options, prefix=init_state_prefix, activ='tanh'))
        init_memory.append(get_layer('ff')[1](tparams, cnn_features, options, prefix=init_memory_prefix, activ='tanh'))

    print 'Building f_init...',
    f_init = theano.function([ctx, cnn_features], [ctx0]+init_state+init_memory, name='f_init', profile=False, allow_input_downcast=True)
    print 'Done'

    # build f_next
    x = tensor.vector('x_sampler', dtype='int64')
    init_state = []
    init_memory = []
    for lidx in range(options['n_layers_lstm']):
        init_state.append(tensor.matrix('init_state', dtype='float32'))
        init_memory.append(tensor.matrix('init_memory', dtype='float32'))

    # for the first word (which is coded with -1), emb should be all zero
    emb = tensor.switch(x[:,None] < 0, 
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                        tparams['Wemb'][x])

    next_state, next_memory, ctxs = [], [], []
    for lidx in range(options['n_layers_lstm']):
        decoder_prefix = 'decoder_%d'%lidx if lidx > 0 else 'decoder'
        inps = proj_h if lidx > 0 else emb
        proj = get_layer('lstm_cond')[1](tparams, inps, options,
                                         prefix=decoder_prefix,
                                         context=ctx0,
                                         one_step=True,
                                         init_state=init_state[lidx],
                                         init_memory=init_memory[lidx],
                                         trng=trng,
                                         use_noise=use_noise)
        next_state.append(proj[0])
        next_memory.append(proj[1])
        ctxs.append(proj[4])
        next_alpha = proj[2]
        proj_h = proj[0]

    if options['use_dropout']:
        proj_h = dropout_layer(proj[0], use_noise, trng)
    else:
        proj_h = proj[0]
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs[-1], options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    f_next = theano.function([x, ctx0]+init_state+init_memory, [next_probs, next_sample, next_alpha]+
                             next_state+next_memory, name='f_next', profile=False, allow_input_downcast=True)

    return f_init, f_next

def gen_sample_ensemble(f_init, f_next, ctx, cnn_feats, options,
                        trng=None, k=1, maxlen=30):
    # assert the f_init and f_next to be lists
    assert len(f_init) == len(f_next)

    sample = []
    sample_score = []
    sample_alpha = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k, 'float32')
    hyp_states = []
    hyp_memories = []
    hyp_alphas = [[] for _ in range(k)]

    # only matters if we use lstm encoder
    rval = [f(ctx, cnn_feats) for f in f_init]
    ctx0 = [r[0] for r in rval]
    next_state = [[r[1].reshape((1, rval[0][1].shape[-1]))] for r in rval]
    next_memory = [[r[2].reshape((1, rval[0][2].shape[-1]))] for r in rval]
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)
    # next_w = -1 * numpy.ones((1,)).astype('int64')
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = [f(*([next_w, c]+s+m)) for f,s,m,c in zip(f_next, next_state, next_memory, ctx0)]
        next_p = avlist([r[0] for r in rval])
        next_alpha = avlist([r[2] for r in rval])
        # extract all the states and memories
        next_state = [r[3] for r in rval]
        next_memory = [r[4] for r in rval]

        cand_scores = hyp_scores[:,None] - numpy.log(next_p)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)] # (k-dead_k) numpy array of with min nll

        voc_size = next_p.shape[1]
        # indexing into the correct selected captions
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat] # extract costs from top hypothesis

        # a bunch of lists to hold future hypothesis
        new_hyp_samples = []
        new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
        new_hyp_alphas = []
        new_hyp_states = [[] for _ in range(len(f_init))]
        new_hyp_memories = [[] for _ in range(len(f_init))]
        # get the corresponding hypothesis and append the predicted word
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis
            new_hyp_alphas.append(hyp_alphas[ti] + [next_alpha[ti]])
            for eidx in range(len(f_init)):
                new_hyp_states[eidx].append(copy.copy(next_state[eidx][ti]))
                new_hyp_memories[eidx].append(copy.copy(next_memory[eidx][ti]))
            
        # check the finished samples for <eos> character
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = [[] for _ in range(len(f_init))]
        hyp_memories = [[] for _ in range(len(f_init))]

        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                sample_alpha.append(new_hyp_alphas[idx])
                dead_k += 1 # completed sample!
            else:
                new_live_k += 1 # collect collect correct states/memories
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_alphas[idx] = new_hyp_alphas[idx]
                for eidx in range(len(f_init)):
                    hyp_states[eidx].append(new_hyp_states[eidx][idx])
                    hyp_memories[eidx].append(new_hyp_memories[eidx][idx])
        hyp_scores = numpy.asarray(hyp_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = numpy.array([w[-1] for w in hyp_samples])
        next_state = [[numpy.array(hyp_states[eidx])] for eidx in range(len(f_init))]
        next_memory = [[numpy.array(hyp_memories[eidx])] for eidx in range(len(f_init))]

    # dump every remaining one
    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    return sample, sample_score, sample_alpha

# generate sample
def gen_sample(f_init, f_next, ctx, cnn_feats, options,
               trng=None, k=1, maxlen=30):
    """Generate captions with beam search.

    Uses layer definitions and functions defined by build_sampler

    This function uses the beam search algorithm to conditionally
    generate candidate captions. Supports beamsearch.
    Parameters
    -sl---------
    f_init : theano function
        input: annotation, output: initial lstm state and memory
        (also performs transformation on ctx0 if using lstm_encoder)
    f_next: theano function
        takes the previous word/state/memory + ctx0 and runs one
        step through the lstm
    ctx0 : numpy array
        annotation from convnet, of dimension #annotations x # dimension
        [e.g (30 x 300)]
    options : dict
        dictionary of flags and options
    trng : random number generator
    k : int
        size of beam search
    maxlen : int
        maximum allowed caption size

    Returns
    -------
    sample : list of list
        each sublist contains an (encoded) sample from the model
    sample_score : numpy array
        scores of each sample
    """
    sample = []
    sample_score = []
    sample_alpha = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    hyp_memories = []
    hyp_alphas = [[] for _ in range(k)]
    
    # only matters if we use lstm encoder
    rval = f_init(ctx, cnn_feats)
    ctx0 = rval[0]
    next_state = []
    next_memory = []
    # the states are returned as a: (dim,) and this is just a reshape to (1, dim)
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([1, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([1, next_memory[-1].shape[0]])
    # reminder: if next_w = -1, the switch statement
    # in build_sampler is triggered -> (empty word embeddings)
    # next_w = -1 * numpy.ones((1,)).astype('int64')
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        # our "next" state/memory in our previous step is now our "initial" state and memory
        rval = f_next(*([next_w, ctx0]+next_state+next_memory))
        next_p = rval[0]
        next_w = rval[1]
        next_alpha = rval[2]

        # extract all the states and memories
        next_state = []
        next_memory = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[3+lidx])
            next_memory.append(rval[3+options['n_layers_lstm']+lidx])

        cand_scores = hyp_scores[:,None] - numpy.log(next_p)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)] # (k-dead_k) numpy array of with min nll

        voc_size = next_p.shape[1]
        # indexing into the correct selected captions
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat] # extract costs from top hypothesis

        # a bunch of lists to hold future hypothesis
        new_hyp_samples = []
        new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
        new_hyp_states = []
        new_hyp_alphas = []
        for lidx in xrange(options['n_layers_lstm']):
            new_hyp_states.append([])
        new_hyp_memories = []
        for lidx in xrange(options['n_layers_lstm']):
            new_hyp_memories.append([])

        # get the corresponding hypothesis and append the predicted word
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx]) # copy in the cost of that hypothesis
            new_hyp_alphas.append(hyp_alphas[ti] + [next_alpha[ti]])

            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))
        
        # check the finished samples for <eos> character
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        for lidx in xrange(options['n_layers_lstm']):
            hyp_states.append([])
        hyp_memories = []
        for lidx in xrange(options['n_layers_lstm']):
            hyp_memories.append([])

        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                sample_alpha.append(new_hyp_alphas[idx])
                dead_k += 1 # completed sample!
            else:
                new_live_k += 1 # collect collect correct states/memories
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_alphas[idx] = new_hyp_alphas[idx]
                for lidx in xrange(options['n_layers_lstm']):
                    hyp_states[lidx].append(new_hyp_states[lidx][idx])
                for lidx in xrange(options['n_layers_lstm']):
                    hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
        hyp_scores = numpy.array(hyp_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = numpy.array([w[-1] for w in hyp_samples])
        next_state = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(numpy.array(hyp_states[lidx]))
        next_memory = []
        for lidx in xrange(options['n_layers_lstm']):
            next_memory.append(numpy.array(hyp_memories[lidx]))

    # dump every remaining one
    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    return sample, sample_score, sample_alpha

def validate_options(options):
    # Put friendly reminders here
    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')

    if options['lstm_encoder']:
        warnings.warn('Note that this is a 1-D bidirectional LSTM, not 2-D one.')

    if options['use_dropout_lstm']:
        warnings.warn('dropout in the lstm seems not to help')

    return options
