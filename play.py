from __future__ import division
from random import sample
import random
import numpy as np
from numpy import diag
import cPickle
import pickle
import time
import contextlib
import os
import sys
import scipy
from scipy import misc
from sknn.mlp import Regressor, Layer, Convolution
from PIL import Image, ImageEnhance
from lasagne import layers as lasagne, nonlinearities as nl
import glob

os.chdir('/Users/Thomas/git/thesis/newgames')
newest = max(glob.iglob('*.[Pp][Nn][Gg]'), key=os.path.getctime)
img = Image.open(newest).resize((dim*3,dim*3), Image.ANTIALIAS).convert('LA')
bright = ImageEnhance.Brightness(img)
img = bright.enhance(1.5)
contrast = ImageEnhance.Contrast(img)
img = contrast.enhance(3)
img.save('small_game.png')
s.newgimg = scipy.misc.imread('small_game.png', flatten=True)
#os.remove('images5/small_game.png')    

#based on new image, create matrix of tries, return image with 'NN' max val
s.actions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
#keep track of moves in game:
if new==False: 
    s.actions = [v for i, v in enumerate(s.actions) if i not in set(s.played)]
    for p in s.played:
        s.newgimg[p[0]*dim-dim:p[0]*dim, p[1]*dim-dim:p[1]*dim] = s.O
    #PUT BLACK SQUARE WHERE PLAYED PREVIOUSLY
else: 
    s.played = []

    #TEMPORARY:
    s.actions = [v for i, v in enumerate(s.actions) if i not in set([2,4])]
    s.newgimg[dim-dim:dim, 3*dim-dim:3*dim] = s.O
    s.newgimg[dim-dim:dim, dim-dim:dim] = s.O

s.tries = np.empty([9*dim**2,])
for a in s.actions:
    x = a[0]+1 ; y = a[1]+1
    s.newgimg[x*dim-dim:x*dim, y*dim-dim:y*dim] = s.O
    s.tries = np.vstack((s.tries, s.newgimg.flatten()))
    s.newgimg[x*dim-dim:x*dim, y*dim-dim:y*dim] = s.N
s.tries = np.delete(s.tries, 0, 0)
s.max = s.learner.nn_pred(s.tries).argmax()
s.played.append(int(s.max))

print s.learner.nn_pred(s.tries)
s.action = s.actions[s.max]

img = Image.open(newest).resize((150,150), Image.ANTIALIAS)
img.save(newest)
s.game = scipy.misc.imread(newest, flatten=True)
s.robotplayed = s.learner.imvec('/Users/Thomas/git/thesis/robotplayed.png')
s.a = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)][s.max]
x = s.a[0]+1 ; y = s.a[1]+1

s.game[x*50-50:x*50, y*50-50:y*50] = s.robotplayed
s.deflat = np.reshape(s.game, (-1, 150))
scipy.misc.imsave('robotmove.png', s.deflat)
os.chdir('/Users/Thomas/git/thesis')