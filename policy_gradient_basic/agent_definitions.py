

import numpy as np 

class Agent:
	# def __init__(self, env, learning_rate=0.001, gamma=0.99, eps=False):
	def __init__(self, env, learning_rate, gamma, eps):
		self.lr = learning_rate
		self.gamma = gamma
		self.eps = eps
		self.env = env
		self.num_action = self.env.action_space.n
		self.num_states = self.env.observation_space.shape[0]

		self.episode_trajectory = []
		self.iter_info = []

		self.states = []
		self.actions = []
		self.action_probs = []
		self.rewards = []

	def restartEp(self):
		self.iter_info = []

		self.states = []
		self.actions = []
		self.action_probs = []
		self.rewards = []
		self.cummulative_reward = 0


class softmaxAgent(Agent):

	def __init__(self, env, learning_rate=0.001, gamma=0.99, eps=False):
		super().__init__(env, learning_rate, gamma, eps)
		# Agent specific initialisation
		self.cummulative_reward = 0
		self.pp = np.random.randn(self.num_action, self.num_states + 1)		# add bias


	def selectAction(self, deterministic=False):
		# Policy function
		action_distr = np.exp(np.asarray(np.dot(self.pp, np.append(self.state, 1)), dtype=np.float128))
		action_distr /= sum(action_distr)+np.finfo(np.float).eps
		# Action selection
		if self.eps == False and deterministic == False:
		# stochastic policy
			return action_distr, np.random.choice(self.num_action, p=list(action_distr))
		elif deterministic == True:
			return action_distr, np.argmax(action_distr)
		else:
		# epsilon greedy
			if np.random.random() < self.eps:
				return action_distr, np.random.randint(0, self.num_action)
			else:
				return action_distr, np.argmax(action_distr) #, max(action_distr)

	def updatePolicy_episodic(self):
		grad_pp = np.zeros(self.pp.shape)  
		# Calculate episodic gradient update
		for i in range(len(self.states)):
			# Calculate characteristic eligibility
			common_deriv = np.tile(np.append(self.states[i], 1.), (self.num_action,1)) * self.action_probs[i].reshape(-1,1)
			onehot = np.zeros((self.num_action,1))
			onehot[self.actions[i]] = 1
			deriv = np.tile(np.append(self.states[i], 1.), (self.num_action,1)) * onehot - common_deriv
			# Calculate reward contribution with baseline
			total_return = sum([self.gamma **i * rew for i, rew in enumerate(self.rewards[i:])])
			advantage   = total_return - np.mean(self.rewards[i:])
			# Calculate gradient
			grad_pp += deriv * advantage
		# Normalise gradient
		grad_pp /= len(self.states)
		# Update policy parameters
		self.pp -= self.lr * grad_pp 
		# print(self.lr * grad_pp)
		# print(self.pp)


class gaussianAgent(Agent):

	def __init__(self, env, learning_rate=0.001, gamma=0.99, eps=False):
		super().__init__(env, learning_rate, gamma, eps)
		# Agent specific initialisation
		self.cummulative_reward = 0
		self.pp = np.random.randn(2)		# add bias

	def selectAction(self, deterministic=False):
		pass

	def updatePolicy_episodic(self):
		pass