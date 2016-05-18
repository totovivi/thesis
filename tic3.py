














#GAIN TIME:
#KEEP MEMORY OF NN WEIGHTS, AND SET CONVERGENCE RULE

#TAKING SUBSETS OF IMAGES IS COMMON PRACTICE



#READ FROM LARGE IMAGE, LEARN ON SMALL, OUTPUT LARGE












import numpy
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

os.chdir('/Users/Thomas/git/thesis')

class State(numpy.ndarray):
    symbols = {0: "_", 1: "X", 2: "O"}
    
    #3x3 array of zeros
    def __new__(subtype): 
        arr = numpy.zeros((3,3), dtype=numpy.int8)
        return arr.view(subtype)

    def __hash__(s): 
        flat = s.ravel() #array as vector
        code = 0
        for i in xrange(9): code += pow(3,i) * flat[i] #pow(3,i) = 3^i
        return code

    def won(s, player): #all possibilities of winning
        x = s == player
        return numpy.hstack( (x.all(0), x.all(1), diag(x).all(), diag(x[:,::-1]).all()) ).any()

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
        s.laststate_hash = None
        s.player = player
        s.gamehist = []
        s.traced = False
        
        if player == 1:
            if alpha == None: 
                s.alpha = 0.2
            else: 
                s.alpha = alpha
            if epsilon == None: 
                s.epsilon = 0.1
            else: s.epsilon = epsilon
        else: 
            if alpha == None: 
                s.alpha = 0.2
            else: s.alpha = alpha
            if epsilon == None: 
                s.epsilon = 0.1
            else: s.epsilon = epsilon
       
    def enum_actions(s, state):
        #enumerate all possible actions from given state
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
        if num == 1: v = random.sample([s.imvec('X.png'), s.imvec('X2.png'), s.imvec('X3.png'), s.imvec('X4.png'), s.imvec('X5.png')], 1)
        elif num == 2: v = random.sample([s.imvec('O.png'),s.imvec('O2.png'),s.imvec('O3.png'),s.imvec('O4.png'), s.imvec('O5.png')], 1)
        else: v = random.sample([s.imvec('N.png')], 1)
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

    def sig(s, x, deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))


    def neural_train(s, X, y):
        np.random.seed(1)
        # randomly initialize our weights with mean 0
        w0 = 2*np.random.random((X.shape[1], X.shape[0])) - 1
        w1 = 2*np.random.random((X.shape[0],1)) - 1
        for j in xrange(10):
            #guesses at levels 1 & 2
            l1 = s.sig(np.dot(X, w0))
            l2 = s.sig(np.dot(l1, w1))
            #if error small and confidence large, small updates
            delta2 = (y - l2) * s.sig(l2, deriv=True)
            delta1 = delta2.dot(w1.T) * s.sig(l1, deriv=True)
            #weights update
            w0 += X.T.dot(delta1)
            w1 += l1.T.dot(delta2)
        return (w0, w1, l2)

    def neural_test(s, X, w0, w1):
        l1 = s.sig(np.dot(X, w0))
        return s.sig(np.dot(l1, w1))

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

        #Find the actions with the highest value

        #maxval = nn_predict()

        valuemap.sort(key=lambda x:x[0], reverse=True)
        maxval = valuemap[0][0]
        valuemap = filter(lambda x: x[0] >= maxval, valuemap)
        #randomize over the max value actions and return one of them 
        opt = sample(valuemap,1)[0]

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
            s.valuefunc[s.laststate_hash] = (1.0-s.alpha) * s.valuefunc[s.laststate_hash] + s.alpha * val

            #NEW FUNCTION
            #s.valuefunc[s.laststate_hash] = s.alpha * (val - s.valuefunc[s.laststate_hash])

        #update laststate value
        s.lastlaststate_hash = s.laststate_hash
        s.laststate_hash = hash(state)
        #append previous state to the game history
        s.gamehist.append(s.laststate_hash)
        
    def reset(s):
        #reset the class except valuefunction
        #basically start new game but keep values you already updated
        s.laststate_hash = None
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

    def __call__(s, pi, pj): 
        j = pi - 1 #take the first coordinate of the previous state
        i = pj - 1 #take the second coordinate of the previous state
        if s.state[j,i] == 0:
            s.state[j,i] = 1 #mark cell as played by human, PROBABLY NOT NECESSARY TO INPUT THE GAME, JUST SHOW PICTURE



            #import one image per move
            #s.newgimg = s.learner.imvec('newgame.png')

            #based on new image, create matrix of tries, return image with 'NN' max val
            #s.act = s.learner.enum_actions(s.state)
            #s.tries = np.empty([22500,])
            #for a in s.act:
            #    x = a[0]+1 ; y = a[1]+1
            #    s.newgimg[x*50-50:x*50, y*50-50:y*50] = s.O
                #save images into list, later return which is max val
            #    s.tries = np.vstack((s.tries, s.newgimg.flatten()))
            #    s.newgimg[x*50-50:x*50, y*50-50:y*50] = s.N
            #s.tries = np.delete(s.tries, 0, 0)
            #np.savetxt('tries.txt', s.tries)



            #learner.next will not be function of s.state, but of s.tries
            s.learner.next(s.state)

            #learner.next_action with NN
            #record which action led to max
            #display new game image
            #scipy.misc.imsave('newgame.png', s.MAXVAL)

        else:
            print "Wrong entry"
        print s.state #,hash(s.state)

        if s.state.full() or s.state.won(1) or s.state.won(2):
            if s.state.won(1):
                print "Won"
            elif s.state.won(2):
                print "Lost"
            else:
                print "Draw"
            s.reset() #reset the game 

    def selfplay(s, n=10000):
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

    def __init__(s, learner = None, other = None ):
        # No learner argument --> Create Learner Class for Player 2 
        if learner == None:
            s.learner = Learner(player=2)
        # If learner class is passed assign it to learner object   
        else:
            s.learner = learner

        if other == None: #oponent player 
            s.other = Learner(player=1)
        else: 
            s.other = other
            
        s.i = 0

        s.wining = []
        s.valuesave = dict()

        s.X = np.empty([22500,])
        s.y = np.empty([1,])

    def reset(s):
        s.state = State()
        s.learner.reset()
        s.other.reset()

    def play(s):
        s.reset()

        while True: # Update states of both players 
            s.other.next(s.state)    
            s.learner.next(s.state)




            #LEARN WITH GENERATED IMAGES

            #initiate game as image, matrix of observations, nb plays
            s.imgstate = s.learner.fullimg(s.state).flatten()
            s.X = np.vstack((s.X, s.imgstate))
            s.y = np.vstack((s.y, 0.0))
            s.plays = s.X.shape[0]

            if s.learner.lastlaststate_hash == None and s.plays-1 <= 1:
                s.X = np.delete(s.X, 0, 0)
                s.y = np.delete(s.y, 0, 0)

            #input value of game if previous game existed
            if s.learner.lastlaststate_hash != None and s.plays-1 > 1:
                s.y[s.plays-2,] = s.learner.valuefunc[s.learner.lastlaststate_hash]



            #generate image observations of all possible next moves
            s.act = s.learner.enum_actions(s.state)
            s.tries = np.empty([22500,])
            for a in s.act:
                s.state[a] = 2
                s.tries = np.vstack((s.tries, s.learner.fullimg(s.state).flatten()))
                s.state[a] = 0
            s.tries = np.delete(s.tries, 0, 0)

            #NN predict on s.tries, pick max
            #np.savetxt('tries.txt', s.tries)





            if s.state.full() or s.state.won(1) or s.state.won(2):
                s.i += 1

                if s.i % 100 == 0:
                    print s.state
                    np.savetxt('imgstate.txt', np.hstack((s.X, s.y)))
                    #s.w0, s.w1, s.l2 = s.learner.neural_train(s.X, s.y)
                    #print np.hstack((s.y, s.l2))
                    #print abs(s.y - s.l2).mean()
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
    
    p1 = Learner(player = 1, alpha = 0.2, epsilon = 0.1)
    p2 = Learner(player = 2, alpha = 0.2, epsilon = 0.1)
    g = Game(learner = p2, other = p1)

g.selfplay(1200)