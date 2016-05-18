import numpy as np
import random
import scipy
from scipy import misc

D = np.genfromtxt('/Users/Thomas/Dropbox/DS/thesis/data.txt', skip_footer=1)
for i in range(0,2000):
	mat = np.reshape(D[i,0:225], (-1, 15))
	val = D[i,225]
	name = '/Users/Thomas/git/thesis/DOESimgEQval/'+str(i)+'_'+str(val)+'.png'
	scipy.misc.imsave(name, mat)