# Reference - https://www.geeksforgeeks.org/bellman-equation/

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

possible_states = 12
possible_actions = 4

ROW_LEN = 4
COL_LEN = 3

q = np.matrix(np.zeros((possible_states, possible_actions)))
r = np.matrix([	
    [	-1	,	0	,	0	,	-1	]	, #0
	[	0	,	-1	,	0	,	-1	]	, #1
	[	0	,	0	,	1	,	-1	]	, #2
	[	0	,	-1	,	-1	,	-1	]	, #3
	[	-1	,	0	,	-1	,	0	]	, #4
	[	0	,	0	,	0	,	0	]	, #5
	[	-1	,	0	,	-1	,	0	]	, #6
	[	0	,	0	,	-1	,	1	]	, #7
	[	-1	,	-1	,	0	,	0	]	, #8
	[	0	,	-1	,	0	,	-1	]	, #9
	[	0	,	-1	,	0	,	0	]	, #10
	[	0	,	-1	,	0	,	-1	]	]) #11


def get_next_pos(curr_pos,dir):
    if (dir == 0 and curr_pos % ROW_LEN != 0):
        return curr_pos - 1
    elif (dir == 1 and curr_pos + ROW_LEN < (ROW_LEN * COL_LEN)):
        return curr_pos + ROW_LEN
    elif (dir == 2 and (curr_pos % ROW_LEN) != (ROW_LEN-1) and curr_pos < (ROW_LEN*COL_LEN) 
          ):
        return curr_pos + 1
    elif (dir == 3 and curr_pos - ROW_LEN >= 0):
        return curr_pos - ROW_LEN
    else: 
        return -1
		
path = []
def run(epidodes, training = False, start_pos = 0 ):
    rewards = 0
    for i in range(epidodes) :
        state = start_pos # setting init state of robot
        while  state !=3 and state != 5 and state != 7 and state != -1:
            if training == True:
                action = int(np.random.choice([0,1,2,3], 1))
            else :
                action = np.argmax(q[state, :])

            next_state = get_next_pos(state, action)

            if training == False:
                #print(state, action, next_state)
                path.append(state)

            q[state, action] = np.max(r[state, action] + 0.9 * (q[next_state, :]))
            
            state = next_state

            if state == 3: 
                rewards = rewards +1 

        if training == False:      
            path.append(state)

    print(path)

    if training ==True:
        print("Rewards received:" , rewards)
    
    


if __name__ == '__main__':
    print("hello")
    state = 0
    print("Training:")
    run(1000, True)
    print("Testing:")
    run(1, False, 11)


    

