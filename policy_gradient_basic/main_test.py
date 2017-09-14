'''
'''
# import numpy as np 
import signal
import sys
import gym
from gym import envs
import matplotlib.pyplot as plt

from agent_definitions import *

# def sigmoid(x):
# 	return 1 / (1 + np.exp(-x))

def printRes():
	env.reset()
	while True:
		env.render()
		action_prob, action = agent.selectAction(deterministic=True)
		agent.state, reward, done, info = env.step(action)
		if done:
			break
	plt.plot(rews)
	plt.show()

def signal_handler(signal, frame):
	printRes()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

### Environment options
env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('LunarLander-v2')
# env = gym.make('Acrobot-v1')

agent = softmaxAgent(env, learning_rate=0.001, gamma=0.99, eps=False)


num_episodes = 20000
rews = []

for ep in range(num_episodes):
	agent.restartEp()
	agent.state = env.reset()
	num_iter = 0
	while True:
		num_iter += 1
		# env.render()
		# select action
		action_prob, action = agent.selectAction()
		# interact with env
		agent.state, reward, done, info = env.step(action)
		# save trajectory
		agent.states.append(agent.state)
		agent.actions.append(action)
		agent.action_probs.append(action_prob)
		agent.rewards.append(reward)
		agent.cummulative_reward += reward
		# agent.updatePolicy_online()

		if done:
			if ep%1000 == 0 and ep>1:
				print("Episode {} finished after {} iterations, with reward {}.".format(ep, num_iter, np.mean(rews)))
				
				if agent.eps > 0:
					print(agent.eps)
					agent.eps *= agent.eps

			# print("Episode {} finished after {} iterations.".format(ep+1, num_iter))
			agent.updatePolicy_episodic()
			rews.append(agent.cummulative_reward)
			break

env.close()
printRes()
