import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
#q = []
#q = np.zeros((64, 4))

## Reference You tube tutorial - https://www.youtube.com/watch?v=ZhoIgo3qqLU

def run (episodes, is_training = True, render = False):
    env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery=False, render_mode = 'human' if render else "None")

    if(is_training == True):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("frozen_lake8x8.pkl", "rb")
        q = pickle.load(f)
        f.close() 
    
    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor
    
    epsilon = 1  # 1 means 100% random actions
    epsilon_decay_rate = 0.0001  # epsilon decay rate. 1/0.0001 = 10,000

    rng = np.random.default_rng()
    rewards_per_episodes = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] # state: 0 to 63, starting at 0 = top left
        terminated = False # True, when fall in hole or reached goal
        truncated = False  # True, when actions are > 200
    
        while(not terminated and not truncated):
            if(is_training and rng.random() < epsilon):
                action = env.action_space.sample() # 0 - left, 1 - down, 2 - right, 3 - top
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if(is_training == True):
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
            #print(state, action, q[state, action], reward)
            state = new_state
    
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon ==0):
            learning_rate_a = 0.0001

        if(reward ==1):
            rewards_per_episodes[i] = 1
            #print("got reward")

    #print(rewards_per_episodes)

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episodes[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('fozen_lake8x8.png')

    if(is_training == True):
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q,f)
        f.close() 

    print(q)  

if __name__ == '__main__':
    #run(1, is_training= False, render=True)
    #print(q)
    run(1, is_training= False, render=True)
