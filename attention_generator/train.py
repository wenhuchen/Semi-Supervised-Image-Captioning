import theano
import theano.tensor as tensor
import theano.sandbox.cuda 
import _init_paths
import cPickle as pkl
import numpy
import copy
import os
import os.path as osp
import time
import argparse
import json
from optimizers import *
from util import zipp, unzip, itemlist, load_params, init_tparams, HomogeneousData, seqs2words, read_pkl
from cocoEvaluation import cocoEvaluation
from capgen import get_dataset, init_params, \
    build_model, build_sampler, gen_sample, validate_options
from generate_caps import process_examples, collapse

"""Note: all the hyperparameters are stored in a dictionary model_options (or options outside train).
   train() then proceeds to do the following:
       1. The params are initialized (or reloaded)
       2. The computations graph is built symbolically using Theano.
       3. A cost is defined, then gradient are obtained automatically with tensor.grad :D
       4. With some helper functions, gradient descent + periodic saving/printing proceeds
"""
def train(dim_word=300,  # word vector dimensionality
          ctx_dim=300,  # context vector dimensionality
          semantic_dim=300,
          dim=1000,  # the number of LSTM units
          cnn_dim=4096, # CNN feature dimension
          n_layers_att=1,  # number of layers used to compute the attention weights
          n_layers_out=1,  # number of layers used to compute logit
          n_layers_lstm=1,  # number of lstm layers
          n_layers_init=1,  # number of layers to initialize LSTM at time 0
          lstm_encoder=True,  # if True, run bidirectional LSTM on input units
          prev2out=False,  # Feed previous word into logit
          ctx2out=False,  # Feed attention weighted ctx into logit
          cutoff=10,
          patience=5,
          max_epochs=30,
          dispFreq=500,
          decay_c=0.,  # weight decay coeff
          alpha_c=0.,  # doubly stochastic coeff
          lrate=1e-4,  # used only for SGD
          selector=False,  # selector (see paper)
          maxlen=30,  # maximum length of the description
          optimizer='rmsprop',
          pretrained='',
          batch_size = 256,
          saveto='model',  # relative path of saved model file
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq updates
          embedding='../Data/GloVe/vocab_glove.pkl',
          cnn_type='vgg',
          prefix='../Data',  # path to find data
          dataset='coco',
          criterion='Bleu_4',
          switch_test_val=False,
          use_cnninit=True,
          use_dropout=True,  # setting this true turns on dropout at various points
          use_dropout_lstm=False,  # dropout on lstm gates
          save_per_epoch=False): # this saves down the model every epoch

    # hyperparam dict
    model_options = locals().copy()
    model_options = validate_options(model_options)

    # reload options
    if os.path.exists('%s.pkl'%saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print "Using the following parameters:"
    print model_options

    print 'Loading data'
    load_data, prepare_data = get_dataset(model_options['dataset'])

    # Load data from data path
    if 'switch_test_val' in model_options and model_options['switch_test_val']:
        train, valid, worddict = load_data(path=osp.join(model_options['prefix'], model_options['dataset']), 
                                            options=model_options, load_train=True, load_test=True)
    else:
        train, valid, worddict = load_data(path=osp.join(model_options['prefix'], model_options['dataset']), 
                                            options=model_options, load_train=True, load_val=True)

    # Automatically calculate the update frequency
    validFreq = len(train[0])/model_options['batch_size']
    print "Validation frequency is %d"%validFreq

    word_idict = {vv:kk for kk, vv in worddict.iteritems()}
    model_options['n_words'] = len(worddict)
    
    # Initialize (or reload) the parameters using 'model_options'
    # then build the Theano graph
    print 'Building model'
    params = init_params(model_options)
    # Initialize it with glove
    if 'VCemb' in params:
        params['VCemb'] = read_pkl(model_options['embedding']).astype('float32')

    # If there is a same experiment, don't use pretrained weights
    if os.path.exists('%s.npz'%saveto):
        print "Reloading model"
        params = load_params('%s.npz'%saveto, params)
    elif pretrained != '':
        params = load_params(pretrained, params, False) # Only pretrain the Language model

    # numpy arrays -> theano shared variables
    tparams = init_tparams(params)

    # In order, we get:
    #   1) trng - theano random number generator
    #   2) use_noise - flag that turns on dropout
    #   3) inps - inputs for f_grad_shared
    #   4) cost - log likelihood for each sentence
    #   5) opts_out - optional outputs (e.g selector)
    trng, use_noise, \
          inps, alphas,\
          cost, \
          opt_outs = \
          build_model(tparams, model_options)


    # Load evaluator to calculate bleu score
    evaluator = cocoEvaluation(model_options['dataset'])

    # To sample, we use beam search: 1) f_init is a function that initializes
    # the LSTM at time 0 [see top right of page 4], 2) f_next returns the distribution over
    # words and also the new "initial state/memory" see equation
    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # we want the cost without any the regularizers
    # define the log probability
    f_log_probs = theano.function(inps, -cost, profile=False,
                                updates=None, allow_input_downcast=True)

    # Define the cost function + Regularization
    cost = cost.mean()
    # add L2 regularization costs
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # Doubly stochastic regularization
    if alpha_c > 0.:
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = sum([alpha_c*((1.-alpha.sum(0))**2).sum(0).mean() for alpha in alphas])
        cost += alpha_reg

    # Backprop!
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    # to getthe cost after regularization or the gradients, use this

    # f_grad_shared computes the cost and updates adaptive learning rate variables
    # f_update updates the weights of the model
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(model_options['optimizer'])(lr, tparams, grads, inps, cost)

    print 'Optimization'
    train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=model_options['maxlen'])

    # history_bleu is a bare-bones training log, reload history
    history_bleu = []
    if os.path.exists('%s.npz'%saveto):
        history_bleu = numpy.load('%s.npz'%saveto)['history_bleu'].tolist()
    start_epochs = len(history_bleu)
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(start_epochs, model_options['max_epochs']):
        n_samples = 0

        print 'Epoch ', eidx

        for caps in train_iter:
            n_samples += len(caps)
            uidx += 1
            # turn on dropout
            use_noise.set_value(1.)

            # preprocess the caption, recording the
            # time spent to help detect bottlenecks
            pd_start = time.time()
            x, mask, ctx, cnn_feats = prepare_data(caps,
                                                train[1],
                                                train[2],
                                                worddict,
                                                model_options)
            pd_duration = time.time() - pd_start

            if x is None:
                print 'Minibatch with zero sample under length ', model_options['maxlen']
                continue

            # get the cost for the minibatch, and update the weights
            ud_start = time.time()
            cost = f_grad_shared(x, mask, ctx, cnn_feats)

            print "Epoch %d, Updates: %d, Cost is: %f" % (eidx, uidx, cost)

            f_update(model_options['lrate'])
            ud_duration = time.time() - ud_start # some monitoring for each mini-batch

            # Numerical stability check
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'PD ', pd_duration, 'UD ', ud_duration

            # Print a generated sample as a sanity check
            if numpy.mod(uidx, model_options['sampleFreq']) == 0:
                # turn off dropout first
                use_noise.set_value(0.)
                x_s = x
                mask_s = mask
                ctx_s = ctx
                # generate and decode the a subset of the current training batch
                for jj in xrange(numpy.minimum(10, len(caps))):
                    sample, score, alphas = gen_sample(f_init, f_next, ctx_s[jj], cnn_feats[jj], model_options,
                                               trng=trng, maxlen=model_options['maxlen'])
                    # Decode the sample from encoding back to words
                    print 'Truth ',jj,': ',
                    print seqs2words(x_s[:,jj], word_idict)
                    for kk, ss in enumerate([sample[0]]):
                        print 'Sample (', kk,') ', jj, ': ',
                        print seqs2words(ss, word_idict)

            # Log validation loss + checkpoint the model with the best validation log likelihood
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                
                # Do evaluation on validation set
                imgid = collapse([elem[-1] for elem in valid[0]])
                caps = process_examples([f_init], [f_next], imgid, valid[1], valid[2], word_idict, model_options)
                folder = osp.join('../output', '%s_%s'%(saveto, 'val'))
                if not osp.exists(folder):
                    os.mkdir(folder)
                with open(osp.join(folder, 'captions_val2014_results.json'), 'w') as f:
                    json.dump(caps, f)
                eva_result = evaluator.evaluate(folder, False)
                if model_options['criterion'] == 'combine':
                    history_bleu.append(eva_result['Bleu_4']+eva_result['CIDEr'])
                else:
                    history_bleu.append(eva_result[model_options['criterion']])

                # the model with the best validation long likelihood is saved seperately with a different name
                if uidx == 0 or history_bleu[-1] == max(history_bleu):
                    best_p = unzip(tparams)
                    print 'Saving model with best validation ll'
                    params = copy.copy(best_p)
                    params = unzip(tparams)
                    numpy.savez(saveto+'_bestll', history_bleu=history_bleu, **params)
                    bad_counter = 0

                # abort training if perplexity has been increasing for too long
                if len(history_bleu) > model_options['patience'] and history_bleu[-1] <= max(history_bleu[:-model_options['patience']]):
                    bad_counter += 1
                    if bad_counter > model_options['patience']:
                        print 'Early Stop!'
                        estop = True
                        break

                print ' BLEU-4 score ', history_bleu[-1]

            # Checkpoint
            if numpy.mod(uidx, model_options['saveFreq']) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = copy.copy(best_p)
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_bleu=history_bleu, **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

        print 'Seen %d samples' % n_samples

        if estop:
            break

        if model_options['save_per_epoch']:
            numpy.savez(saveto + '_epoch_' + str(eidx + 1), history_bleu=history_bleu, **unzip(tparams))

    # use the best nll parameters for final checkpoint (if they exist)
    if best_p is not None:
        zipp(best_p, tparams)
    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, history_bleu=history_bleu, **params)

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser("Running the training of attention model")
    parser.add_argument("--saveto", type=str, required=True, help="The file to save the model")
    parser.add_argument("--dataset", type=str, required=True, help="Which Dataset")
    parser.add_argument("--gpu", type=int, default=0, help="Select a gpu to use")
    parser.add_argument("--cutoff", type=int, required=True, help="cutoff for the visual detected words")
    parser.add_argument("--pretrained", type=str, default='', help="Embedding")
    parser.add_argument("--region", action='store_true', help="Which Dataset")
    parser.add_argument("--no_review", action='store_true', help="Which Dataset")
    parser.add_argument("--no_cnninit", action='store_true', help="Which Dataset")
    parser.add_argument("--alpha_reg", action='store_true', help="Regularize alpha value or not")
    parser.add_argument("--switch", action='store_true', help="Switch test and validation set")
    parser.add_argument("--cnn_type", type=str, default='vgg', help="Which CNN architecture for image representation")
    args = parser.parse_args()
    
    # Select a gpu to use
    theano.sandbox.cuda.use('gpu%d'%args.gpu)
    cnn_type_mapping = {'googlenet':1024, 'resnet':2048, 'vgg':4096}
    common_kwargs = {'dataset':args.dataset, 'saveto':args.saveto,'cnn_type': args.cnn_type, 
                    'lstm_encoder': not args.no_review, 'pretrained': args.pretrained, 
                    'use_cnninit': not args.no_cnninit,
                    'switch_test_val': args.switch,
                    'semantic_dim': 2048 if args.region else 300, 
                    'alpha_c': 0.01 if args.alpha_reg else 0,
                    'batch_size': 128 if 'flickr' in args.dataset else 256, 
                    'cutoff': 10 if args.region else args.cutoff, 
                    'cnn_dim': cnn_type_mapping[args.cnn_type]}

    train(**common_kwargs)
