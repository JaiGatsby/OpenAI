######################################
""" AI Final Project
	Solve Lunar Lander

	Group 2-
	SRA Jai Singh
	JADOON Asfandayar Khan
"""
######################################

# Import statements
import gym 
import numpy as np
import cPickle as pickle


env = gym.make('LunarLander-v2') # Loads the environment details

# Hyperparameters
noise_scaling = 0.8 
bestreward = -9999 # initialized to large -ve value
num_episodes = 5000
episodes_per_update = 10
updates_to_render = 1

# Initialize weights randomly
parameters = np.random.rand(8, 4) * 2 - 1

rList = [] # Stores rewards oer episode

## Generates a new set of weights given existing weights and adds noise based on noise
def getParams(p,slow):
	global noise_scaling
	if slow:
		noise_scaling = 0.1
	return p + (np.random.rand(8, 4) * 2 - 1)*noise_scaling

## Runs one episode of the actor trying out the environment and returns the reward
def run_episode(env, parameters,render):  
	observation = env.reset() # Resets the env to a new start position
	totalreward = 0
	i = 0
	while True:
		i +=1
		if render: # will explicitly show the last updates_to_render*episodes_per_update runs, usefull for debugging
			env.render()

		# Chooses an action based on weighted sum of observations and then finding the argmax
		action = np.argmax(np.matmul(observation,parameters))

		# The action is taken and the environment returns some observations and rewards
		observation, reward, done, _ = env.step(action)

		# adds rewards per instance to sum for the entire episode
		totalreward += reward

		# provided by OpenAI, it encodes when the agent has won/died
		if done:
			break
	return totalreward

## Returns the average reward gained by a set of weights
def update(env,np,render):
	r = 0
	for _ in xrange(episodes_per_update):
		run = run_episode(env,np,render)
		r+= run/float(episodes_per_update)
	return r
 

for x in xrange(num_episodes):  
	newparams = getParams(parameters,x>2000) # switches the noise_scaling after the 2000th episode
	reward = 0
	reward = update(env,newparams,num_episodes-x < updates_to_render)
	rList.append(reward)

	# Debug statements
	# if x % 100 == 0 and x>=100:
	# 	print str(x/200)+" BestRewardSoFar " + str(bestreward)
	# 	print (sum(rList[x-100:x])//100)

	# updates the weights
	if reward > bestreward:
		bestreward = reward
		parameters = newparams

env.close()

# stores some data for easier accessibility in the future as well as analytics
store = [parameters,rList,bestreward]
pickle.dump(store, open("savedata.p", "wb"))