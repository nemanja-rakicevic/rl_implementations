
import numpy as np 
import gym

class Agent:
	# def __init__(self, env, learning_rate=0.001, gamma=0.99, eps=False):
	def __init__(self, env, learning_rate, gamma, eps):
		self.lr = learning_rate
		self.gamma = gamma
		self.eps = eps
		self.env = env
		# get action dimensions
		if type(env.action_space) == gym.spaces.Discrete:
			self.num_action = self.env.action_space.n
		elif type(env.action_space) == gym.spaces.Box:
			self.num_action = self.env.action_space.shape[0]
		# get state dimensions
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
	'''
	This agent calculates the discreete action probabilities by applying a 
	linear projection of the current state, and squeezing the output through a 
	softmax function.
	'''
	def __init__(self, env, learning_rate=0.001, gamma=0.99, eps=False):
		super().__init__(env, learning_rate, gamma, eps)
		# Agent specific initialisation
		self.cummulative_reward = 0
		self.pp = np.random.randn(self.num_action, self.num_states + 1)		# add bias

	def selectAction(self, deterministic=False):
		current_state = np.append(self.state, 1)
		# Policy function
		action_distr = np.exp(np.asarray(np.dot(self.pp, current_state), dtype=np.float128))
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
		grad_pp = np.zeros_like(self.pp)  
		# Calculate episodic gradient update
		num_iter = len(self.states)
		for i in range(num_iter):
			# Calculate reward contribution with baseline
			total_return = sum([self.gamma **i * rew for i, rew in enumerate(self.rewards[i:])])
			advantage   = total_return - np.mean(self.rewards[i:])

			# Calculate characteristic eligibility
			common_deriv = np.tile(np.append(self.states[i], 1.), (self.num_action,1)) * self.action_probs[i].reshape(-1,1)
			onehot = np.zeros((self.num_action,1))
			onehot[self.actions[i]] = 1
			deriv = np.tile(np.append(self.states[i], 1.), (self.num_action,1)) * onehot - common_deriv
			# Calculate gradient
			grad_pp += deriv * advantage
		# Normalise gradient updates
		grad_pp /= num_iter
		# Update policy parameters
		self.pp -= self.lr * grad_pp 



class gaussianAgent(Agent):
	'''
	This agent calculates the continuous action values through the Gaussian function.
	The input of the Gaussian is a linear projection of the current state,
	and the mean and standard deviation values are separately updated parameters.
	'''
	def __init__(self, env, learning_rate=0.001, gamma=0.99, eps=False):
		super().__init__(env, learning_rate, gamma, eps)
		# Agent specific initialisation
		self.cummulative_reward = 0
		# define parameter matrix
		self.pp = [[],[],[]]		
		# parameters linear
		self.pp[0] = np.random.randn(self.num_states + 1, self.num_action)		# add bias
		# parameters mean
		self.pp[1] = np.random.randn(self.num_action)
		# parameters sigma
		self.pp[2] = np.random.randn(self.num_action)

	def selectAction(self, deterministic=False):
		current_state = np.append(self.state, 1)
		y = np.dot(self.pp[0].T, current_state)
		actions = np.exp(-((y - self.pp[1])**2)/(2*self.pp[2]**2)) / (np.sqrt(2*np.pi)*self.pp[2])
		
		return actions, actions

	def updatePolicy_episodic(self):
		grad_pp0 = np.zeros_like(self.pp[0]) 
		grad_pp1 = np.zeros_like(self.pp[1]) 
		grad_pp2 = np.zeros_like(self.pp[2]) 
		# Calculate episodic gradient update
		num_iter = len(self.states)
		for i in range(num_iter):
			# Calculate reward contribution with baseline
			total_return = sum([self.gamma **i * rew for i, rew in enumerate(self.rewards[i:])])
			advantage    = total_return - np.mean(self.rewards[i:])

			# Calculate characteristic eligibility 
			current_state = np.append(self.states[i], 1)
			y = np.dot(self.pp[0].T, current_state)

			# parameter linear
			eligib_0 = ((self.pp[1] - y)/self.pp[2]**2) * np.tile(current_state.reshape(-1,1), self.num_action)
			grad_pp0 += eligib_0 * advantage
			# parameter mean
			eligib1 = (y - self.pp[1])/self.pp[2]**2
			grad_pp1 += eligib1 * advantage
			# parameter sigma
			eligib2 = ((y - self.pp[1])**2 - self.pp[2]**2)/self.pp[2]**3
			grad_pp2 += eligib2 * advantage
		# Normalise gradient updates
		grad_pp0 /= num_iter
		grad_pp1 /= num_iter
		grad_pp2 /= num_iter
		# Update parameters
		self.pp[0] -= self.lr * grad_pp0
		self.pp[1] -= self.lr*self.pp[2]**2 * grad_pp1
		self.pp[2] -= self.lr*self.pp[2]**2 * grad_pp2 

def fcn_softmax(x):
	softmax = np.exp(np.asarray(x, dtype=np.float128))
	softmax /= sum(softmax)+np.finfo(np.float).eps
	return softmax

def fcn_sigmoid(x):
	return 1. / (np.exp(np.asarray(-x, dtype=np.float128)) + 1.)

def deriv_tanh(x):
	return 1 - np.tanh(x)**2

def deriv_softmax(x, idx):
	common_deriv = np.tile(fcn_softmax(x)[idx], len(x)) * fcn_softmax(x)
	onehot = np.zeros_like(x)
	onehot[idx] = 1
	deriv = np.tile(fcn_softmax(x)[idx], len(x))  * onehot - common_deriv
	return deriv
		
def deriv_sigmoid(x):
	return fcn_sigmoid(x)*(1 - fcn_sigmoid(x))


class mlpAgent(Agent):
	'''
	This agent calculates the actions by applying a 
	multilayer perceptron (input-hidden-output) with a tanh activation function in the hidden layer,
	and a softmax output layer for discreete actions, 
	or a tanh output layer for continuous actions
	'''
	def __init__(self, env, learning_rate=0.001, gamma=0.99, eps=False, num_hidden=10):
		super().__init__(env, learning_rate, gamma, eps)
		# Agent specific initialisation
		self.num_hidden = num_hidden
		self.cummulative_reward = 0
		# define parameter matrix
		self.pp = [[],[]]		
		# parameters linear
		self.pp[0] = np.random.randn(self.num_states + 1, self.num_hidden)		# W1
		# parameters mean
		self.pp[1] = np.random.randn(self.num_hidden + 1, self.num_action)		# W2

		self.fcn_activation = fcn_sigmoid
		self.deriv_activation = deriv_sigmoid

	def forwardPass(self, current_state):
		# input current_state
		# hidden
		hidden = self.fcn_activation(np.dot(self.pp[0].T, current_state))
		hidden = np.append(hidden, 1)
		# output
		out = fcn_softmax(np.dot(self.pp[1].T, hidden))
		return out, hidden

	def selectAction(self, deterministic=False):
		current_state = np.append(self.state, 1)
		action_distr, _ = self.forwardPass(current_state)

		action_distr[-1] = 1 - action_distr[:-1].sum()	# numerical stuff: make the probabilities sum to 1

		return action_distr, np.random.choice(self.num_action, p=list(action_distr))

	def updatePolicy_episodic(self):

		grad_pp0 = np.zeros_like(self.pp[0]) 
		grad_pp1 = np.zeros_like(self.pp[1])  

		# Calculate episodic gradient update
		num_iter = len(self.states)

		for i in range(num_iter):

			# Calculate reward contribution with baseline
			total_return = sum([self.gamma **i * rew for i, rew in enumerate(self.rewards[i:])])
			advantage   = total_return - np.mean(self.rewards[i:])
			
			# Calculate characteristic eligibility
			current_state = np.append(self.states[i], 1)
						
			# W1 update
			output, hidden = self.forwardPass(current_state)
			eligib0 = (1/output) * \
					  deriv_softmax(np.dot(self.pp[1].T, hidden), self.actions[i]) * \
					  self.pp[1][:-1,:] * \
					  self.deriv_activation(np.dot( self.pp[0].T, current_state)).reshape(-1,1)
			eligib0 = np.outer(current_state, eligib0.sum(axis=1))


			# W2 update
			eligib1 = (1/output) * \
					  deriv_softmax(np.dot(self.pp[1].T, hidden), self.actions[i]) * \
					  np.tile(hidden.reshape(-1,1),2)

			# Calculate gradient
			grad_pp0 += eligib0 * advantage
			grad_pp1 += eligib1 * advantage
		
		# Normalise gradient updates
		grad_pp0 /= num_iter
		grad_pp1 /= num_iter
		# Update policy parameters
		self.pp[0] -= self.lr * grad_pp0
		self.pp[1] -= self.lr * grad_pp1


