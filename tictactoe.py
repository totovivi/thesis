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

os.chdir('/Users/Thomas/Dropbox/DS/thesis/')

#I THINK USELESS
#@contextlib.contextmanager
#def ignore_stderr():
    #devnull = os.open(os.devnull, os.O_WRONLY)
    #old_stderr = os.dup(2)
    #sys.stderr.flush()
    #os.dup2(devnull, 2)
    #os.close(devnull)
    #try:
        #yield
    #finally:
        #os.dup2(old_stderr, 2)
        #os.close(old_stderr)

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

class Dog:
    kind = 'canine'         
    def __init__(self, name):
        self.name = name    

dog = Dog('Yoyo')
print dog.kind

class Image:
    #for test, returns image, after will return vector 
    def __init__(s,):
        #not State(), returns zeros

    def imvec(img):
        return scipy.misc.imread(img, flatten=True)

    def numvec(num):
        if num == 1: v = random.sample([imvec('X.png'), imvec('X2.png'), imvec('X3.png'), imvec('X4.png'), imvec('X5.png')], 1)
        elif num == 2: v = random.sample([imvec('O.png'), imvec('O2.png'), imvec('O3.png'), imvec('O4.png'), imvec('O5.png')], 1)
        else: v = random.sample([imvec('N.png')], 1)
        return v

    def stateobs():
        obs = []
        for i in xrange(3):
            for j in xrange(3):
                obs.append(numvec(State[i,j]))
        return obs

    def fullimg():
        row1 = np.hstack((stateobs(s.state)[0][0], stateobs(s.state)[1][0], stateobs(s.state)[2][0]))
        row2 = np.hstack((stateobs(s.state)[3][0], stateobs(s.state)[4][0], stateobs(s.state)[5][0]))
        row3 = np.hstack((stateobs(s.state)[6][0], stateobs(s.state)[7][0], stateobs(s.state)[8][0]))
        return np.vstack((row1, row2, row3))

print Image.fullimg
scipy.misc.imsave('outfile.png', Image.fullimg)
        
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
                s.epsilon = 0.02
            else: s.epsilon = epsilon
        else: 
            if alpha == None: 
                s.alpha = 0.2 
            else: s.alpha = alpha
            if epsilon == None: 
                s.epsilon = 0.02
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
        #return list of all possible actions
        return res

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
            elif state.full(): val = 0.0
            #else, game continues
            else: val = 0.0
            #assign value to the new state
            s.valuefunc[hashval] = val
        #reset state to the old value    
        state[action] = 0
        #return value of the new state
        return val

    #def nn_train():
        #if s.laststate_hash == None:
            #train = numpy.array(X, t)
        #train[X].append(board_img_vec(s))
        #train[t].append(s.valuefunc[s.laststate_hash])
        #fit = nn(X, t, data = train)
        #return fit

    #def nn_predict():
        #pred = []
        #for i in 1:9:
            #pred.append(predict(nn_train(), tryi))
        #return which gave max(pred)
        
    def next_action(s, state): #decide action after maximizing gain
        valuemap = list()
        #enumerate over all possible actions
        for action in s.enum_actions(state):
            #check value of the new state if you make a possible action
            val = s.value(state, action)
            #add it to value map associated with the given action
            valuemap.append( (val, action) )

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
        #If the other player won assign value -1
        if state.won(3-s.player):
            val = -1
        #If the game ended assign value -0.1
        elif state.full():
            val = -0.1
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
        #update laststate value
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
    
    #define the reset function    
    def reset(s):
        s.state = State()
        ##########################
        s.img = Image() #CHANGE
        ##########################
        s.learner.reset() 
        print s.state

    def __call__(s, pi, pj): 
        j = pi -1 #take the first coordinate of the previous state
        i = pj - 1 #take the second coordinate of the previous state
        if s.state[j,i] == 0:
            s.state[j,i] = 1 #mark cell as played by human
            s.learner.next(s.state)
        else:
            print "Wrong entry"
        print s.state #,hash(s.state)
        #############
        s.img #CHANGE
        #############

        if s.state.full() or s.state.won(1) or s.state.won(2):
            if s.state.won(1):
                print "Won"
            elif s.state.won(2):
                print "Lost"
            else:
                print "DRAW"
            s.reset() #reset the game 

    def selfplay(s, n=10000):
        #selfplay for specific number of rounds
        for i in xrange(n):
            s.sp.play() 
        s.reset() #in the end reset again
    #use the package cPicle to save the dictionary 
    def save(s):
        cPickle.dump(s.learner, open("learn.dat", "w")) #open is for the appointing the name file 
        # w = write, we can even add wb so that it is portable between Windows and Unix 
    #use the package cPickle to load the dictionary 
    def load(s):
        s.learner = cPickle.load( open("learn.dat") ) #read the dictionary
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

        if other == None:
        # Create oponent player 
            s.other = Learner(player=1)
        else: 
            s.other = other
            
        # Set counter to zero
        s.i = 0

        s.wining = []
        s.valuesave = dict()


    def reset(s):
        # Create state class
        s.state = State()
        # Reset both players 
        s.learner.reset()
        s.other.reset()

    def play(s):
        # Initliaze states and players
        s.reset()

        while True:
            # Update states of both players 
            s.other.next(s.state)    
            s.learner.next(s.state)

            # Check if board is full or if player 1 or 2 won 
            if s.state.full() or s.state.won(1) or s.state.won(2):
                # FALSE: Update counter
                s.i += 1
                # In every 100th iteration print the current state
                if s.i % 100 == 0:
                    print s.state
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
    
    p1 = Learner(player = 1, alpha = 0.2, epsilon = 0.02)
    p2 = Learner(player = 2, alpha = 0.2, epsilon = 0.02)
    g = Game(learner = p2,other=p1)