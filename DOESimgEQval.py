import numpy as np
import random
import scipy
from scipy import misc
import scipy.misc
from PIL import Image, ImageEnhance

D = np.genfromtxt('/Users/Thomas/Dropbox/DS/thesis/data.txt', skip_footer=1)
for i in range(0,100):
	mat = np.reshape(D[i,0:81], (-1, 9))
	val = D[i,81]
	name = '/Users/Thomas/git/thesis/DOESimgEQval/'+str(i)+'_'+str(val)+'.png'
	scipy.misc.toimage(mat, cmin=0, cmax=255).save(name)