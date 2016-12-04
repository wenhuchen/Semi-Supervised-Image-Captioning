import theano
import theano.tensor as tensor
from util import ortho_weight, norm_weight, tanh, rectifier, linear, _p
import numpy

def param_init_mlp(options, params, prefix='predictor'):
    dims = options['dims']
    layer_num = len(dims)
    assert layer_num >= 3
    for i in range(layer_num-1):
        W = norm_weight(dims[i], dims[i+1])
        params[_p(prefix, i)] = W
    return params

def param_init_attention(options, params, prefix='attention'):
    dim_word = options['dim_word']
    params[_p(prefix, 'Wm')] = norm_weight(dim_word)
    params[_p(prefix, 'b')] = numpy.zeros((dim_word,), dtype='float32')
    params[_p(prefix, 'W_att')] = norm_weight(dim_word)
    params[_p(prefix, 'U_att')] = norm_weight(dim_word, 1)
    params[_p(prefix, 'c_att')] = numpy.zeros((1,), dtype='float32') 
    return params

def mlp_layer(tparams, state_below, options, prefix='predictor'):
    layer_num = len(options['dims'])
    for i in range(layer_num-1):
        if i == 0:
            output = tensor.dot(state_below, tparams[_p(prefix, i)])
            output = tanh(output)
        elif i == layer_num - 2:
            output = tensor.dot(output, tparams[_p(prefix, i)])
            output = rectifier(output)
        else:
            output = tensor.dot(output, tparams[_p(prefix, i)])
            output = tanh(output)
    return output

def mlp_attention_layer(tparams, state_below, options, prefix='attention'):
    mean_emb = state_below.mean(1)
    attention_vec = tensor.dot(state_below, tparams[_p(prefix, 'W_att')]) + tparams[_p(prefix, 'b')] 
    attention_vec += tensor.dot(mean_emb, tparams[_p(prefix, 'Wm')])[:, None, :]
    attention_vec = tanh(attention_vec)
    alpha = tensor.dot(attention_vec, tparams[_p(prefix, 'U_att')]) + tparams[_p(prefix, 'c_att')]
    alpha_shp = alpha.shape
    alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0], alpha_shp[1]]))
    output = (state_below * alpha[:, :, None]).sum(1)
    return output
