#for layers: tensorflow.org/versions/r0.8/api_docs/python/nn.html


import theano
theano.config.optimizer='fast_compile'
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import os

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    #pyx = softmax(T.dot(l4, w_o))
    pyx = T.nnet.sigmoid(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

data_dir = os.path.join('/Users/Thomas/git/thesis')
fd = open(os.path.join(data_dir,'X.txt'))
loaded = np.loadtxt(fd)
trX = loaded.reshape((937,15*15)).astype(float)

fd = open(os.path.join(data_dir,'y.txt'))
loaded = np.loadtxt(fd)
trY = loaded.reshape((937)).astype(float)

trX = trX.reshape(-1, 1, 15, 15)

X = T.TensorType(dtype='float64', broadcastable=(False, True, False, False))()
Y = T.fvector()

w = init_weights((20, 1, 3, 3))
w2 = init_weights((40, 20, 3, 3))
w3 = init_weights((1, 40, 3, 3))
w4 = init_weights((1, 60))
w_o = init_weights((60, 1))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)

cost = T.mean(T.sqr(py_x - Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 1), range(1, len(trX), 1)):
        cost = train(trX[start:end], trY[start:end])
    #print np.mean((trY - predict(trX)).sqrt())
    print predict(trX)[800:820]
    print trY[800:820]
    print 'yoyo'