#S1: vector of pixels including robot 1 and robot 2's first choices
#we append a matrix of states, with corresponding targets from the V() function
#we hope the NN will discover patterns

ROUND 1: V(S1) = 0 #impossible to win or loose with one move, nothing to approximate and choose from neural network
ROUND 2: V(S1) = V(S1) + alpha(val - V(S1)) #still nothing, val = observed value|option with maximum expected value was chosen
               = 0 + alpha(0 - 0)
ROUND 3: V(S2) = V(S2) + alpha(val - V(S1)) 
#with no previous info, our random play will lead val = -1, we get
               = 0 + alpha(-1 - 0) = -alpha 

#NEW GAME: same as previous by chance and image of game strictly identical
ROUND 1: V(S4) = V(most probable state = S1) = 0 #the NN should not confuse exact similar vectors with another one, returns V(S1). With time it will identify better starting points
#NN must identify which undelrying state we update, here will choose S2 (we have to set a confidence rule, if none exceed confidence, new state)
ROUND 2: V(S2) = V(S4) + alpha(val - V(S4)) #S5 = S2, NN evaluates all possible next states: so if move results in S2, prediction = (proba1)*(-alpha), otherwise, prediction = (proba~0)*(-alpha or 0). We pick the second option which results in
			   = 0 + alpha(0 - 0)
ROUND 3: V(S5) = V(S5) + alpha(val - V(S5))


ROUND 3: V(S6) = V(S2) + alpha(val - V(S1)) 
S3 = #one option will be estimated as (NN_confidence = 1)*(NN_prediction = -alpha), the other option will be estimated as (NN_confidence ~ 0)*(NN_prediction = -alpha or 0)
#after picking the new option, we get val = 1
	 S2 + alpha(val - S2) = 0 + alpha(1 - 0) = alpha
#this should differentiate well final 2 from final option 1






#V(S_t-1) = alpha(val|choice of state maximizing E[V] - V(S_t-1))
#choice of state maximizing E[V] = state.max(pred(NN, put cross in option 1), pred(NN, put cross in option 2)...)
#V(S_t-1) will feed the matrix used for the inference, then the predictions