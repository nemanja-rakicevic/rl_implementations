l'''
'''

import numpy as np 
import gym
from gym import envs


# def sigmoid(x):
# 	return 1 / (1 + np.exp(-x))

class Agent:
	def __init__(self, env, learning_rate=0.01):
		self.lr = learning_rate
		self.gamma = 0.95
		self.env = env
		self.num_action = self.env.action_space.n
		self.num_states = self.env.observation_space.shape[0]
		self.state = np.zeros(self.num_states)
		self.cummulative_reward = 0
		self.pp = np.random.randn(self.num_action, self.num_states + 1)		# add bias
		self.states = []
		self.actions = []
		self.rewards = []

	def restartEp(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.cummulative_reward = 0


class linearAgent(Agent):

	def selectAction(self):
		action_distr = np.dot(self.pp, np.append(self.state, 1))
		# action_distr /= sum(action_distr)
		return action_distr, np.argmax(action_distr) #, max(action_distr)

	def updatePolicy_episodic(self):
		grad_pp = np.zeros(self.pp.shape)  

		for i in range(len(self.states)):
			
			deriv = np.hstack((np.vstack((self.states[i], self.states[i])), [[1.],[1.]]))/np.tile(self.actions[i].reshape(len(self.actions[i]),1), len(self.states[i])+1)
			# deriv = np.hstack((np.vstack((self.states[i], self.states[i])), [[1.],[1.]]))
			
			total_return = sum([self.gamma **i * rew for i, rew in enumerate(self.rewards)])
			
			advantage   = total_return - np.mean(self.rewards)
			
			grad_pp += deriv * advantage


		grad_pp /= len(self.states)

		self.pp -= self.lr * grad_pp 

		# print(self.lr * grad_pp)
		# print(self.pp)


num_episodes = 1000
env = gym.make('CartPole-v1')
agent = linearAgent(env)

for ep in range(num_episodes):
	agent.restartEp()
	agent.state = env.reset()
	num_iter = 0
	while True:
		num_iter += 1
		env.render()
		# select action
		action_prob, action = agent.selectAction()
		# interact with env
		agent.state, reward, done, info = env.step(action)
		# save trajectory
		agent.states.append(agent.state)
		agent.actions.append(action_prob)
		agent.rewards.append(reward)
		agent.cummulative_reward += reward
		# agent.updatePolicy_online()

		if done:
			# print("Episode {} finished after {} iterations, with reward {}.".format(ep+1, num_iter, agent.cummulative_reward))
			print("Episode {} finished after {} iterations".format(ep+1, num_iter))
			
			agent.updatePolicy_episodic()
			break

env.close()