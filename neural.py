import numpy as np

def sig(x, deriv=False):
	if(deriv==True):
	    return x*(1-x)
	return 1/(1+np.exp(-x))

def neural_train(X, y):

	np.random.seed(1)

	# randomly initialize our weights with mean 0
	w0 = 2 * np.random.random((X.shape[1], X.shape[0])) - 1
	w1 = 2 * np.random.random((X.shape[0],1)) - 1

	for j in xrange(100):

		#guesses at levels 1 & 2
	    l1 = sig(np.dot(X, w0))
	    l2 = sig(np.dot(l1, w1))
	        
	    #if error small and confidence large, small updates
	    delta2 = (y - l2) * sig(l2, deriv=True)
	    delta1 = delta2.dot(w1.T) * sig(l1, deriv=True)

	    w0 += X.T.dot(delta1)
	    w1 += l1.T.dot(delta2)

	print 'l1',l1.shape
	print 'l2',l2.shape
	print 'delta1',delta1.shape
	print 'delta2',delta2.shape
	print 'w1',l1.shape
	print 'w0',w0.shape

	return (w0, w1, l2)

def neural_test(X, w0, w1):
	l1 = sig(np.dot(X, w0))
	return sig(np.dot(l1, w1))

#X = np.array([[0, 2, 1, 5, 30], [0, 2, 4, 5, 15], [1, 1, 3, 5, 5]])
#y = np.array([[1], [1], [.5]])
data = np.loadtxt('/Users/Thomas/git/thesis/imgstate.txt', delimiter=' ')[0:10,:]
X = data[:,0:22499]
y = data[:,22500][np.newaxis].T
w0, w1, pred = neural_train(X, y)
print np.hstack((y, pred))

#print neural_test(X_, w0, w1)