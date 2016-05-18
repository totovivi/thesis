from sknn.mlp import Regressor, Layer
import numpy as np
np.set_printoptions(threshold=100000)
import pickle
import os
import sys
import time
from sknn.platform import cpu64, threading

os.chdir('/Users/Thomas/git/thesis/')

data = np.loadtxt('imgstate.txt', delimiter=' ')

X = data[:,0:224]
Y = data[:,225][np.newaxis].T

if 'params' in locals():
	params = params
else:
	params = None

def neural(X, Y, params=None):
	nn = Regressor(
		layers=[
		    Layer("Rectifier", units=200),
		    Layer("Sigmoid", units=200),
		    Layer("Rectifier", units=200),
		    Layer("Sigmoid", units=200),
		    Layer("Sigmoid")],
		learning_rate=0.001,
		n_iter=100,
		parameters=params,
		learning_rule='sgd',
		f_stable=0.001,
		valid_size=.2,
		verbose=True,
		n_stable=3)

	return nn.fit(X, Y)

nn1 = neural(X, Y)

#model in pkl
pickle.dump(nn1, open('nn.pkl', 'wb'))
neural_loaded = pickle.load(open('nn.pkl', 'rb'))
neural_loaded.predict(X)
params = neural_loaded.get_parameters()

#after loop
nn2 = neural(X, Y, params)











#params in csv
#params = neural.get_parameters()
#wr = csv.writer(open('params.csv', 'wb'), quoting=csv.QUOTE_ALL)
#wr.writerow(params)
#f = csv.reader(codecs.open('params.csv', 'rU'))
#import pandas as pd
#data = pd.read_csv('params.csv')
#with open('params.csv', 'rb') as f:
#    reader = csv.reader(f)



#pred =  neural.predict(x)
#print np.hstack((pred, y))
#print abs(pred - y).mean()