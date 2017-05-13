import gym
import numpy as np
from gym import wrappers

# Setting up the environment
env = gym.make('FrozenLake-v0') # using FrozenLake for now as im having dependency issues and it is a discrete state system
# env = wrappers.Monitor(env, '/tmp/frozenlake-experiment-1')

print(env.action_space)
#Discrete(4)
print(env.observation_space)
#Discrete(16)

qstates = np.zeros([env.observation_space.n,env.action_space.n]) # a 16x4 of zeros, each representing an action per state
# eps = 0.2
alpha = 0.85
gamma = 0.9999

num_episodes = 2500

rList = []
for i_episode in range(num_episodes):
	observation = env.reset() #generates new env, and gives initial observation
	# print("observation",observation)
	rAll = 0
	while True:
		# env.render()
		action = np.argmax(qstates[observation,:] + np.random.randn(1,env.action_space.n)*(1./(i_episode+1)))
		# print("action",action)
		observation1, reward, done, info = env.step(action) #takes action
		# print("observation",observation1)
		# print("reward", reward)
		if done:
			if reward != 0:
				altReward = 20 # Goal
			else:
				altReward = -5 # Dead
		else:
			altReward = -0.00001 #living reward
		qstates[observation,action] = qstates[observation,action] + alpha*(altReward+(gamma*np.max(qstates[observation1,:])) - qstates[observation,action])
		observation = observation1
		rAll += reward
		if done:
			# if (i_episode%500 == 0):
				# print("Episode finished after {} timesteps".format(t+1), "reward: ", rAll)
			break
	rList.append(rAll)

print("Score over time:" + str(sum(rList)/num_episodes))
# print("score over last 100: ", sum(rList[:-100])/100)