import argparse
import numpy
import cPickle as pkl
import os.path as osp
import os
import sys
import json
from capgen import init_params, get_dataset, build_sampler, gen_sample, gen_sample_ensemble
from util import load_params, init_tparams, seqs2words, read_pkl
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
trng = RandomStreams(1234)

def collapse(list_of_id):
    result = []
    for item in list_of_id:
        if len(result) == 0 or item != result[-1]:
            result.append(item)
    return result

# single instance of a sampling process
def gen_model(model, options):

    # this is zero indicate we are not using dropout in the graph
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    # get the parameters
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    # build the sampling computational graph
    # see capgen.py for more detailed explanations
    f_init, f_next = build_sampler(tparams, options, use_noise, trng)

    return f_init, f_next 

def process_examples(f_init, f_next, imgid, contexts, cnn_feats, word_idict, options, k=4, normalize=False, debug=False):
    caps = []
    if len(cnn_feats) < len(contexts):
        for idx, ctx in enumerate(contexts):
            if options['with_glove']:
                ctx_cutoff = ctx[:options['cutoff']*options['semantic_dim']].reshape([options['cutoff'], options['semantic_dim']])
            else:
                ctx_cutoff = ctx[:options['cutoff']]
            if len(f_init) > 1 and len(f_next) > 1:
                sample, score, alpha = gen_sample_ensemble(f_init, f_next, ctx_cutoff, cnn_feats[0], 
                                                        options, trng=trng, k=k, maxlen=30)
            else:
                sample, score, alpha = gen_sample(f_init[0], f_next[0], ctx_cutoff, cnn_feats[0], 
                                                        options, trng=trng, k=k, maxlen=30)
            if normalize:
                lengths = numpy.array([len(s) for s in sample])
                score = score / lengths
            sidx = numpy.argmin(score)
            # write result into caption format
            caps.append({'image_id':imgid[idx],'caption':seqs2words(sample[sidx], word_idict)})
            if idx % 100 == 0:
                print 'Sample %d/%d'%(idx, len(contexts))      
        return caps
    elif len(cnn_feats) == len(contexts):
        for idx, ctx, ctx_cnn in zip(range(len(contexts)), contexts, cnn_feats):
            if options['with_glove']:
                ctx_cutoff = ctx[:options['cutoff']*options['semantic_dim']].reshape((options['cutoff'], options['semantic_dim']))
            else:
                ctx_cutoff = ctx[:options['cutoff']]
            # generate the samples
            if len(f_init) > 1 and len(f_next) > 1:
                sample, score, alpha = gen_sample_ensemble(f_init, f_next, ctx_cutoff, ctx_cnn, 
                                                        options, trng=trng, k=k, maxlen=30)
            else:
                sample, score, alpha = gen_sample(f_init[0], f_next[0], ctx_cutoff, ctx_cnn, 
                                                        options, trng=trng, k=k, maxlen=30)
            if normalize:
                lengths = numpy.array([len(s) for s in sample])
                score = score / lengths
            sidx = numpy.argmin(score)
            # write result into caption format
            caption = seqs2words(sample[sidx], word_idict) 
            caps.append({'image_id':imgid[idx],'caption': caption})
            if idx % 100 == 0:
                print 'Sample %d/%d'%(idx, len(contexts))
            if debug:
                if idx < 6:
                    for word, weights in zip(caption.split(), alpha[sidx]):
                        print word, weights
                    print
                else:
                    quit()
        return caps
    else:
        raise ValueError("The length of cnn features and contexts does not equal")

def main(pkl_names, models, split, k=4, normalize=False, debug=False, changes=None):
    # load model model_options
    f_init, f_next = [], []
    for pkl_name, model in zip(pkl_names, models):
        options = read_pkl(pkl_name)
        if args.changes is not None:
            for change in args.changes:
                options[change.split('=')[0]] = change.split('=')[1]
        # initialize the two functions
        f1, f2 = gen_model(model, options)
        f_init.append(f1)
        f_next.append(f2)
        
    # fetch data, skip ones we aren't using to save time
    load_data, _ = get_dataset(options['dataset'])
    kwargs = {'path':osp.join(options['prefix'], options['dataset']), 'load_%s'%split:True, 'options':options}
    eval_data, worddict = load_data(**kwargs)
    imgid = collapse([elem[-1] for elem in eval_data[0]])
    word_idict = {vv:kk for kk, vv in worddict.iteritems()}
        
    # write results to json format
    caps = process_examples(f_init, f_next, imgid, eval_data[1], eval_data[2], word_idict, options, k, normalize, debug=debug)

    # create folder if not exist
    if len(pkl_names) > 1:
        folder = osp.join('../output', '%s_ensemble_%s'%(options['dataset'], split))
    else:
        folder = osp.join('../output', '%s_%s'%(osp.splitext(pkl_names[0])[0], split))

    # If there exists more, then create mirrows
    if not osp.exists(folder):
        os.mkdir(folder)
    elif osp.exists(folder) and split=='test':
        for i in range(2, 5):
            if not osp.exists('%s.%d'%(folder, i)):
                folder = '%s.%d'%(folder, i)
                os.mkdir(folder)
                break

    # write json to the file
    with open(osp.join(folder, 'captions_val2014_results.json'), 'w') as f:
        json.dump(caps, f)
    
    if split in ('val', 'test'):
        # Evaluate using the official api
        coco_caption_folder = osp.join('../', 'coco-caption')
        assert osp.exists(coco_caption_folder)
        sys.path.append(coco_caption_folder)
        from cocoEvaluation import cocoEvaluation
        evaluator = cocoEvaluation(options['dataset'])
        evaluator.evaluate(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--beam_size', type=int, default=4, help="beam size to use")
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--split', type=str, required=True, default='val')
    parser.add_argument('--pkl_name', type=str, required=True, nargs='+', help="name of pickle file (without the .pkl)")
    parser.add_argument('--model', type=str, required=True, nargs='+', help="The model file")
    parser.add_argument('--salient', type=str, default='', help="The salient words")
    parser.add_argument('--changes', type=str, nargs='+', help="changes to the original pkl")
    args = parser.parse_args()
    
    main(args.pkl_name, args.model, args.split, k=args.beam_size,
        normalize=args.normalize, debug=args.debug, changes=args.changes)
