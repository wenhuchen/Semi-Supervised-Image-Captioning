import theano
import theano.tensor as tensor
from util import ortho_weight, norm_weight, tanh, rectifier, linear
import numpy
from utils import _p

def param_init_lstm_cond_nox(options, params, prefix='lstm_cond_nox', dim=None, dimctx=None):
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    
    # LSTM to LSTM
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # bias to LSTM
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')
   
    # from context to gates
    Wc = norm_weight(dimctx, dim*4)
    params[_p(prefix, 'Wc')] = Wc

    Wc_att = norm_weight(dimctx, ortho=False)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attnetion: last context -> hidden
    Wct_att = norm_weight(dimctx, ortho=False)
    #params[_p(prefix,'Wct_att')] = Wct_att

    Wd_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1, )).astype('float32')
    params[_p(prefix, 'c_att')] = c_att

    return params

def lstm_cond_nox_layer(tparams, options, prefix='lstm_cond_nox',
            context=None, init_state=None, init_memory=None,**kwargs):

    assert context, 'Context must be provided'
    # get the steps of context
    if context.ndim == 2:
        new_ctx = context[None,: , :]
    else:
        new_ctx = context

    dim = tparams[_p(prefix, 'U')].shape[0]
    n_samples = new_ctx.shape[0]
    nsteps = new_ctx.shape[1]
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)
    mask = tensor.alloc(1., nsteps, 1)
    
    # pre-compute projected context
    pctx_ = tensor.dot(new_ctx, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]
    
    def _step(m_, h_, c_, a_, ct_, pctx_, dp_=None, dp_att_=None):
        if _p(prefix, 'Wct_att') in tparams:
            pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')]) + tensor.dot(ct_, tparams[_p(prefix, 'Wct_att')])
        else:
            pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')])
        pctx_ += pstate_[:, None, :]
        pctx_ = tanh(pctx_)
        alpha = tensor.dot(pctx_, tparams[_p(prefix, 'U_att')] + tparams[_p(prefix, 'c_att')])
        alpha_shp = alpha.shape
        alpha = tensor.nnet.softmax(alpha.reshape((alpha_shp[0], alpha_shp[1])))
        ctx_ = (new_ctx * alpha[:,:,None]).sum(1) # current context
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')]) + \
                tensor.dot(ctx_, tparams[_p(prefix, 'Wc')]) + \
                tparams[_p(prefix,'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))

        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:,None] * c_

        h = o*tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:,None] * h_

        rval = [h, c, alpha, ctx_, i, f, o]
        return rval

    outputs_info = [init_state,
                    init_memory,
                    tensor.alloc(0., n_samples, nsteps),
                    new_ctx.mean(1)]
    outputs_info += [None,
                     None,
                     None]
    rval, updates = theano.scan(_step,
                                non_sequences = [pctx_],
                                sequences = [mask],
                                outputs_info=outputs_info,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False)
    if context.ndim == 2:
        return rval[0].squeeze(), rval[2].squeeze()
    else:
        return rval[0], rval[2]
