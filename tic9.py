from __future__ import division














#TASKS:
#   theory for s.y
#   s.y matches the right s.X
#   prediction goes well













from random import sample
import random
import numpy as np
from numpy import diag
import cPickle
import time
import contextlib
import os
import sys
import scipy
from scipy import misc
from sknn.mlp import Regressor, Layer
from PIL import Image, ImageEnhance
from lasagne import layers as lasagne, nonlinearities as nl

smalls = '/Users/Thomas/git/thesis/images2/smalls'
os.chdir(smalls)
N = [scipy.misc.imread('N1.png', flatten=True)]
Xs = []
Os = []
max = 0
pred = 0
who = 'none'
dim=5
for i in os.listdir(smalls):
    if i.startswith('X'):
        Xs.append(scipy.misc.imread(i, flatten=True))
    elif i.startswith('O'):
        Os.append(scipy.misc.imread(i, flatten=True))
os.chdir('/Users/Thomas/git/thesis')

class State(np.ndarray):
    symbols = {0: "_", 1: "X", 2: "O"}
    
    #3x3 array of zeros
    def __new__(subtype): 
        arr = np.zeros((3,3), dtype=np.int8)
        return arr.view(subtype)

    def __hash__(s): 
        flat = s.ravel() #array as vector
        code = 0
        for i in xrange(9): code += pow(3,i) * flat[i] #pow(3,i) = 3^i
        return code

    def won(s, player): #all possibilities of winning
        x = s == player
        return np.hstack( (x.all(0), x.all(1), diag(x).all(), diag(x[:,::-1]).all()) ).any()

    def full(s):
        return (s != 0).all()

    def __str__(s): #fill the board with current state
        out = [""]
        for i in xrange(3):
            for j in xrange(3):
                out.append(s.symbols[s[i,j]]) #we defined the possible symbols at the begining of the class
            out.append("\n")
        return str(" ").join(out)

class Learner:
    def __init__(s, player, alpha = None, epsilon = None):
        s.valuefunc = dict()
        s.X = dict()
        s.y = dict()
        s.laststate_hash = None
        s.player = player
        s.gamehist = []
        s.traced = False
        s.nn = Regressor(
            layers=[
                Layer("Rectifier", units=200),
                Layer("Sigmoid", units=200),
                Layer("Rectifier", units=200),
                Layer("Sigmoid", units=200),
                Layer("Sigmoid")],
            learning_rate=0.001,
            n_iter=200)
        
        if player == 1:
            if alpha == None: 
                s.alpha = 0.8
            else: 
                s.alpha = alpha
            if epsilon == None: 
                s.epsilon = 1
            else: s.epsilon = epsilon
        else: 
            if alpha == None: 
                s.alpha = 0.8
            else: s.alpha = alpha
            if epsilon == None: 
                s.epsilon = 0.1
            else: s.epsilon = epsilon
       
    def enum_actions(s, state):
        #all possible actions from given state
        res = list()
        for i in xrange(3):
            for j in xrange(3):
                #if a given position in the given state
                #is still empty then add it as a possible action 
                if state[i,j] == 0:
                    res.append((i,j))
        return res

    def imvec(s, img):
        return scipy.misc.imread(img, flatten=True)

    def numvec(s, num):
        if num == 1: v = random.sample(Xs, 1)
        elif num == 2: v = random.sample(Os, 1)
        else: v = N
        os.chdir('/Users/Thomas/git/thesis')
        return v

    def stateobs(s, state):
        obs = []
        for i in xrange(3):
            for j in xrange(3):
                obs.append(s.numvec(state[i,j]))
        return obs

    def fullimg(s, state):
        row1 = np.hstack((s.stateobs(state)[0][0], s.stateobs(state)[1][0], s.stateobs(state)[2][0]))
        row2 = np.hstack((s.stateobs(state)[3][0], s.stateobs(state)[4][0], s.stateobs(state)[5][0]))
        row3 = np.hstack((s.stateobs(state)[6][0], s.stateobs(state)[7][0], s.stateobs(state)[8][0]))
        return np.vstack((row1, row2, row3))

    def nn_train(s, X, y):
        return s.nn.fit(X, y)
            
    def nn_test(s, X):
        return s.nn.predict(X)

    def value(s, state, action): #measures the gain after a particular step
        "Assumption: Game has not been won by other player"
        #modify the state: put to the given place(action) the given symbol(player)
        state[action] = s.player
        #hash value is an id used to compare disctionary keys quickly, gives another value to floats (keeps order)
        #id of new state
        hashval = hash(state)
        #access value of the new state
        val = s.valuefunc.get(hashval)

        #if new state has no value yet
        if val == None:
            #if new state is winning assign value 1
            if state.won(s.player): val = 1.0
            #if new state is final but player did not win assign value 0
            elif state.full(): val = 0.5
            #else, game continues
            else: val = 0.2
            #assign value to the new state
            s.valuefunc[hashval] = val
        #reset state to the old value    
        state[action] = 0
        #return value of the new state
        return val
        
    def next_action(s, state): #decide action after maximizing gain
        valuemap = list()
        #enumerate over all possible actions
        for action in s.enum_actions(state):
            #check value of the new state if you make a possible action
            val = s.value(state, action)
            #add it to value map associated with the given action
            valuemap.append((val, action))

        # Random Choice before sorting 
        rc = sample(valuemap, 1)[0]

        valuemap.sort(key=lambda x:x[0], reverse=True)
        maxval = valuemap[0][0]
        valuemap = filter(lambda x: x[0] >= maxval, valuemap)
        #randomize over the max value actions and return one of them 
        opt = sample(valuemap,1)[0]

        if who == 'learner':
            move = s.enum_actions(state)[max]
            #print move
            val = s.value(state, move)
            opt = [val, move]
            global who
            who = 'other'

        split = np.random.choice(2, 1, p=[1-s.epsilon, s.epsilon]).tolist()[0]
        if split == 1:
            return rc
        else:
            return opt

    def next(s, state):
        #If the other player won assign
        if state.won(3-s.player):
            val = 0.0
        #If the game ended assign
        elif state.full():
            val = 0.5
        else:
            #Otherwise find the best action with the associated value
            (val, action) = s.next_action(state)
            #Redefine state according to this action (put the given 
            #player`s sign to the optimal action)
            state[action] = s.player

        if state.won(1) or state.won(2) or state.full():
            #If game is finish change traced value to true
            s.traced = True
            
        #learning step
        #If there was a previous state
        if s.laststate_hash != None:
            #update the value of the previous state (meaning the state you were in the previous step) 
            #based on the original values of the previous states, valuefunc, and the value of the new state, val
            s.valuefunc[s.laststate_hash] = s.valuefunc[s.laststate_hash] + s.alpha * (val - s.valuefunc[s.laststate_hash])

            #Value function for learner
            s.X[s.laststate_hash] = s.fullimg(state).flatten()
            s.y[s.laststate_hash] = val

        s.laststate_learner = s.laststate_hash
        #update laststate value
        s.laststate_hash = hash(state)
        #append previous state to the game history
        s.gamehist.append(s.laststate_hash)
        
    def reset(s):
        #reset the class except valuefunction
        #basically start new game but keep values you already updated
        s.laststate_hash = None
        s.laststate_learner = None
        s.gamehist = []
        s.traced = False
                        
class Game:
    #description of the game = the variable 
    #what objects it should have inside
    def __init__(s, learner = None , other = None ):
        if learner == None: 
            s.learner = Learner(player=2) #define if we want a second player 
        else:
            s.learner = learner
        if other == None:
            s.other = Learner(player=1)
        else:
            s.other = other
        s.reset() #define that reset is part of the game 
        s.sp = Selfplay(s.learner) #if we want self learning
        
        s.O = s.learner.imvec('O.png')
        s.N = s.learner.imvec('N.png')
    
    #define the reset function    
    def reset(s):
        s.state = State()
        s.learner.reset()
        print s.state

    def __call__(s, pi=1, pj=1): #whatever original assignement

        j = pi - 1 #take the first coordinate of the previous state
        i = pj - 1 #take the second coordinate of the previous state
        if s.state[j,i] == 0:
            #import and treat one image per move
            img = Image.open('newgames/new.png').resize((dim*3,dim*3), Image.ANTIALIAS)
            bright = ImageEnhance.Brightness(img)
            #img = bright.enhance(1.8)
            contrast = ImageEnhance.Contrast(img)
            #img = contrast.enhance(40)
            img.save('newgames/new_small.png')
            s.newgimg = scipy.misc.imread('newgames/new_small.png', flatten=True)

            #based on new image, create matrix of tries, return image with 'NN' max val
            s.actions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
            s.tries = np.empty([dim*3*dim*3,])
            for a in s.actions:
                x = a[0]+1 ; y = a[1]+1
                s.newgimg[x*dim-dim:x*dim, y*dim-dim:y*dim] = s.O
                s.tries = np.vstack((s.tries, s.newgimg.flatten()))
                s.newgimg[x*dim-dim:x*dim, y*dim-dim:y*dim] = s.N
            s.tries = np.delete(s.tries, 0, 0)
            s.max = s.learner.nn_test(s.tries).argmax()

            print s.learner.nn_test(s.tries)
            s.action = s.actions[s.max]

            s.game = scipy.misc.imread('newgames/new.png', flatten=True)
            s.robotplayed = s.learner.imvec('robotplayed.png')
            s.a = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)][s.max]
            x = s.a[0]+1 ; y = s.a[1]+1
            s.game[x*50-50:x*50, y*50-50:y*50] = s.robotplayed
            s.deflat = np.reshape(s.game, (-1, 150))
            scipy.misc.imsave('newgames/robotmove.png', s.deflat)

    def selfplay(s, n):
        #selfplay for specific number of rounds
        for i in xrange(n):
            s.sp.play() #seen in load()
        s.reset() #in the end reset again

    def save(s):
        cPickle.dump(s.learner, open("learn.dat", "w")) #open is for the appointing the name file 
        # w = write, we can even add wb so that it is portable between Windows and Unix 

    def load(s):
        s.learner = cPickle.load(open("learn.dat")) #read the dictionary
        s.sp = Selfplay(s.learner) #selfplay using that dictionary 
        s.reset() # reset at the end
             
class Selfplay:

    def __init__(s, learner = None, other = None):
        # No learner argument --> Create Learner Class for Player 2 
        if learner == None:
            s.learner = Learner(player=2)
        # If learner class is passed assign it to learner object   
        else:
            s.learner = learner

        if other == None: 
            s.other = Learner(player=1)
        else: 
            s.other = other
            
        s.i = 0
        s.games = 0
        s.wins = 0

        s.wining = []
        s.valuesave = dict()

        s.X = np.empty([dim*3*dim*3,])
        s.y = np.empty([1,])

    def reset(s):
        s.state = State()
        s.learner.reset()
        s.other.reset()

    def play(s):
        s.reset()


        while True: # Update states of both players 

            #OTHER CHOOSES NEXT MOVE, LEARNER EVALUATES IMAGE EQUIVALENTS THEN CHOOSES NEXT MOVE
            s.other.next(s.state) 

            #generate image observations of all possible next moves
            s.tries = np.empty([dim*3*dim*3,])
            for a in s.learner.enum_actions(s.state):
                s.state[a] = 2
                s.tries = np.vstack((s.tries, s.learner.fullimg(s.state).flatten()))
                s.state[a] = 0
            s.tries = np.delete(s.tries, 0, 0)

            try:
                global who
                who = 'learner'
                global max
                max = s.learner.nn_test(s.tries).argmax()
                global pred
                pred = s.learner.nn_test(s.tries).max()
            except:
                global max
                max = 0
                global pred
                pred = 0

            #HERE IF GAME ENDED, APPEND, ELSE:

            s.learner.next(s.state)

            #ONCE LEARNER PLAYED, LEARN X and y and append

            #

            #PRODUCE N DISTORTED OBSERVATIONS EACH TIME

            #update X and y only from second move
            s.statekey = s.learner.laststate_learner
            #CHECK IF Y AND X ALIGNED
            if s.X.shape[0] == 1:
                s.X[0] = s.learner.X[s.statekey]
                s.y[0] = s.learner.y[s.statekey]
            elif s.statekey != None: #append if not after reset
            #ALSO APPEND ONLY AFTER LEARNER MOVED
                s.X = np.vstack((s.X, s.learner.X[s.statekey]))
                s.y = np.vstack((s.y, s.learner.y[s.statekey]))
            else: 
                pass
            s.plays = s.X.shape[0]

            print s.plays
            print s.state
            print 'played position ', max, 'prediction: ', pred, 'result: ', s.y[len(s.y)-1]
            if s.state.full() or s.state.won(1) or s.state.won(2):
                s.games += 1
            if s.state.won(2):
                s.wins += 1
            print ' '
            print ' '

            #learn from updated matrix every 10 games
            if s.plays % 1 == 0:
                if s.state.full() or s.state.won(1) or s.state.won(2):
                    np.savetxt('imgstate.txt', np.hstack((s.X, s.y)))
                    scipy.misc.imsave('newgame.png', s.learner.fullimg(s.state))
                    print 'wins proportion: ', s.wins/s.games
                    #s.wins = 0
                    #s.games = 0
                    s.learner.nn_train(s.X, s.y)

                    np.save('X', s.X)
                    np.save('y', s.y)



            










            if s.state.full() or s.state.won(1) or s.state.won(2):
                s.i += 1

                #if s.i % 10 == 0:
                    #print s.state
                # If game is not finish do the the optimised next step
                if not s.other.traced:
                    s.other.next(s.state)

                if s.state.won(1): s.wining.append(1)
                elif s.state.won(2): s.wining.append(2)
                else: s.wining.append(0)






                for j in [1,3,9,27,81,243,729,2187,6561]:
                    s.other.valuefunc
                    if s.valuesave.get(j) == None: s.valuesave[j] = [s.other.valuefunc[j]]
                    else: s.valuesave[j].append(s.other.valuefunc[j])
                for j in [7, 19, 163, 13123, 5, 165, 13125, 13203, 567]: 
                    s.learner.valuefunc
                    if s.valuesave.get(j) == None:
                        if s.learner.valuefunc.get(j) == None: s.valuesave[j] = [None]
                        else: s.valuesave[j] = [s.learner.valuefunc[j]]
                    else:
                        if s.learner.valuefunc.get(j) == None: s.valuesave[j].append(None)
                        else: s.valuesave[j].append(s.learner.valuefunc[j])
                        
                break 

if __name__ == "__main__":
    
    p1 = Learner(player = 1)
    p2 = Learner(player = 2)
    g = Game()

g.selfplay(2)

#g()