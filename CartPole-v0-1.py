import gym
# import tensorflow as tf
import numpy as np
from gym import wrappers
env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

# W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# # hypothesis = W*x + b

def sigmoid_array(x):                                        
	return 1 / (1 + np.exp(-x))

def run_episode(env, parameters):  
	observation = env.reset()
	totalreward = 0
	for _ in range(200):
		action = sigmoid_array(np.matmul(parameters,observation))
		observation, reward, done, info = env.step(action)
		totalreward += reward
		if done:
			break
	return totalreward

noise_scaling = 0.1  
parameters = np.random.rand(24,4) * 2 - 1
bestreward = 0
for i in range(1000):  
	newparams = parameters + (np.random.rand(24,4) * 2 - 1)*noise_scaling
	reward = 0
	reward = run_episode(env,newparams)
	if i%500==0:
		print(reward)
	if reward > bestreward:
		bestreward = reward
		parameters = newparams
	if reward == 200:
		print("done",i)
		break