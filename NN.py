import random
import math, arff
import numpy as np

np.seterr(all = 'ignore')

def sigmoid(x):
	return 1/(1 + np.exp(-x))

#derivative of sigmoid
def dsigmoid(y):
	return y * (1.0 - y)

#using tanh instead of logistic sigmoid
def tanh(x):
	return math.tanh(x)

#derivative of tanh
def dtanh(y):
	return 1 - y * y

class MLP_NeuralNetwork(object):
	
	def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
		'''
			input: Number of Input Neurons
			output: Number of Ouput Neurons
			hidden: Number of hidden Neurons
		'''
		self.iterations = iterations
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.rate_decay = rate_decay
		
		#initializing arrays
		self.input = input + 1
		self.output = output
		self.hidden = hidden
		
		#setting array of 1.0's for activation
		self.ai = [1.0] * self.input
		self.ao = [1.0] * self.output
		self.ah = [1.0] * self.hidden
		
		#create randomized weights
		input_range = 1.0/self.input ** (1/2)
		output_range = 1.0/self.hidden ** (1/2)
		
		self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
		self.wo = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, self.output))
		
		#create array of 0's for changes
		self.ci = np.zeros((self.input, self.hidden))
		self.co = np.zeros((self.hidden, self.output))		
		
	
	def feedForward(self, inputs):
		'''
			Feed Forward Propagation
		'''	
		
		if len(inputs) != self.input - 1:
			raise ValueError("Wrong Number of Inputs!")
		
		#input activations
		for i in range(self.input - 1): #-1 because of bias term
			self.ai[i] = inputs[i]
		
		#hidden layer activations
		for i in range(self.hidden):
			sum_ = 0.0
			for j in range(self.input):
				sum_ += self.ai[j] * self.wi[j][i]
			self.ah[i] = tanh(sum_)			#assigning the activation
		
		#output activations
		for k in range(self.output):		
			sum_ = 0.0
			for i in range(self.hidden):
				sum_ += self.ah[i] * self.wi[i][k]
			self.ao[k] = sigmoid(sum_)			#assigning the activation
		
		return self.ao[:]
	
	def backPropagate(self, targets):
		'''
			Back Propagate the error
		'''
		
		if len(targets) != self.output:
			raise ValueError("Wrong Number of Ouptut Values!")
		
		#calculate error factor for output
		output_deltas = [0.0] * self.output
		for i in range(self.output):
			error = -(targets[i] - self.ao[i])
			output_deltas[i] = dsigmoid(self.ao[i]) * error
		
		#calculate error terms for hidden layers
		hidden_deltas = [0.0] * self.hidden
		for i in range(self.hidden):
			error = 0.0
			for j in range(self.output):
				error += output_deltas[j] * self.wo[i][j]
			hidden_deltas[i] = dtanh(self.ah[i]) * error
		
		#update weights connecting hidden to output
		for i in range(self.hidden):
			for j in range(self.output):
				change = output_deltas[j] * self.ah[i]
				self.wo[i][j] -= self.learning_rate * change + self.co[i][j] * self.momentum
				self.co[i][j] = change
		
		#update weights connecting input to hidden
		for i in range(self.input):
			for j in range(self.hidden):
				change = hidden_deltas[j] * self.ai[i]
				self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum
				self.ci[i][j] = change
			
		#calculate error
		error = 0.0		
		for i in range(len(targets)):
			error += 0.5 * (targets[i] - self.ao[i]) ** 2

		return error
		
	def test(self, patterns):		
		'''
			this will print out the targets next to the predictions.
		'''
		for p in patterns:
			print(p[1], '->', self.feedForward(p[0]))
		
	def train(self, patterns):
		'''
			Train our NN
		'''
		for i in range(self.iterations):
			error = 0.0
			random.shuffle(patterns)

			for p in patterns:
				inputs = p[0]
				targets = p[1]				
				
				self.feedForward(inputs)
				error += self.backPropagate(targets)

			with open('error.txt','w') as errorfile:
				errorfile.write(str(error)+'\n')
			errorfile.close()
			
			if i%10 == 0:
				print('error %-.5f'% error)
			
			self.learning_rate = self.learning_rate * (self.learning_rate/(self.learning_rate + self.learning_rate * self.rate_decay))
	
	def predict(self, X):
		'''
			return list of prediction of algorithm
		'''
		predictions = []
		for p in X:
			predictions.append(self.feedForward(p))
		return predictions
	
def demo():
	'''
		Run NN on Network Flow Dataset
	'''

	def parse_output(y):
		output = []
		for j in y:	
			y1 = []
			for i in range(0, 11):
				if i == j-1:
					y1.append(1)
				else:
					y1.append(0)			
			output.append(y1)
			
		return output
	
	def load_data():
		'''
			Load dataset in data object
		'''
		dataset = arff.load(open('NIMS.arff','r'))
		ds = np.array(dataset['data'], dtype = np.float64)
		
		data = ds[0:10001, 0:22]
		y = ds[0:10001, 22]
		
		y = parse_output(y)
			
		X1 = ds[10001:10002,0:22]
		y1 = ds[10001:10002,22]
		
		y1 = parse_output(y1)
		
		X1 -=data.min()
		X1 /=data.max()
		
		#scaling data values to be in between 0 and 1
		data -=data.min()
		data /=data.max()
		
		out = []
		print(data.shape)
		
		#populate the data
		for i in range(data.shape[0]):
			tup = list((data[i, :].tolist(), y[i]))
			out.append(tup)
		
		X2 = []
		for i in range(X1.shape[0]):
			tup = list(X1[i,:].tolist())
			X2.append(tup)
		
		NN = MLP_NeuralNetwork(22, 22, 11, iterations = 50, learning_rate = 0.5, momentum = 0.5, rate_decay = 0.01)
	
		NN.train(out)
		#NN.test(out)
		pdict = NN.predict(X2)
	
		print(pdict[0], y1[0])
	
		return out
	
	X = load_data()
	
	
if __name__ == '__main__':
	demo()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
