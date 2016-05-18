import numpy as np
import scipy
from scipy import misc
import os
import random

#NEURAL NETWORK
def neural_net(X, y):
	def nonlin(x, deriv=False):
		if(deriv==True):
		    return x*(1-x)
		return 1/(1+np.exp(-x))

	np.random.seed(1)

	# randomly initialize our weights with mean 0
	syn0 = 2*np.random.random((X.shape[1],X.shape[0])) - 1
	syn1 = 2*np.random.random((X.shape[0],1)) - 1

	for j in xrange(10000):

		# Feed forward through layers 0, 1, and 2
	    l0 = X
	    l1 = nonlin(np.dot(l0,syn0))
	    l2 = nonlin(np.dot(l1,syn1))

	    # how much did we miss the target value?
	    l2_error = y - l2
	    
	    if (j% 1000) == 0:
	        print "Error:" + str(np.mean(np.abs(l2_error)))
	        
	    # in what direction is the target value?
	    # were we really sure? if so, don't change too much.
	    l2_delta = l2_error*nonlin(l2,deriv=True)

	    # how much did each l1 value contribute to the l2 error (according to the weights)?
	    l1_error = l2_delta.dot(syn1.T)
	    
	    # in what direction is the target l1?
	    # were we really sure? if so, don't change too much.
	    l1_delta = l1_error * nonlin(l1,deriv=True)

	    syn1 += l1.T.dot(l2_delta)
	    syn0 += l0.T.dot(l1_delta)

	print l2



#DATA
State = np.array([[0,0,2],
                  [1,2,1],
                  [2,0,0]])

os.chdir('/Users/Thomas/Dropbox/DS/thesis/')

def imvec(img):
	return scipy.misc.imread(img, flatten=True)
#add flatten here

def numvec(num):
	if num == 1: v = random.sample([imvec('X.png'), imvec('X2.png'), imvec('X3.png'), imvec('X4.png'), imvec('X5.png')], 1)
	elif num == 2: v = random.sample([imvec('O.png'), imvec('O2.png'), imvec('O3.png'), imvec('O4.png'), imvec('O5.png')], 1)
	else: v = random.sample([imvec('N.png')], 1)
	return v

def stateobs(State):
	obs = []
	for i in xrange(3):
	    for j in xrange(3):
	    	obs.append(numvec(State[i,j]))
	return obs

def fullimg():
	row1 = np.hstack((stateobs(State)[0][0], stateobs(State)[1][0], stateobs(State)[2][0]))
	row2 = np.hstack((stateobs(State)[3][0], stateobs(State)[4][0], stateobs(State)[5][0]))
	row3 = np.hstack((stateobs(State)[6][0], stateobs(State)[7][0], stateobs(State)[8][0]))
	return np.vstack((row1, row2, row3))

scipy.misc.imsave('outfile4.png', fullimg())



#APPLY NEURAL NETWORK
x1 = scipy.misc.imread('outfile1.png').flatten()
x2 = scipy.misc.imread('outfile2.png').flatten()
x3 = scipy.misc.imread('outfile3.png').flatten()
x4 = scipy.misc.imread('outfile4.png').flatten()
X = np.vstack((x1, x2, x3, x4))

y = np.array([[0],[0],[1],[1]])

neural_net(X, y)