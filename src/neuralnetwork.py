import numpy as np
import os.path
import pudb

# np.seterr(all='raise')

def sigmoid(x):
	y = (1 / (1 + np.exp(-x)))
	return y, y * (1 - y)

def softplus(x):
	return np.log(1 + np.exp(-x)), 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x), (x >= 0) * 1

def squaredError(layer, targets, isDerivative=False):
	if isDerivative:
		return (layer.outputs - np.array(targets))*layer.derivative
	else:
		return 0.5*np.sum((layer.outputs - np.array(targets))**2)

def crossEntropyError(layer, targets, isDerivative=False):
	if isDerivative:
		return layer.outputs - np.array(targets)
	else:
		np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
		t = np.array(targets)
		y = np.array(layer.outputs)
		return -np.sum(np.nan_to_num(t*np.lon(y) + (1 - t)*np.log(1 - y)))


def save(neuralnetwork, fileName, overwrite=False, printResults=True):
	weights = []
	for layer in neuralnetwork.layers:
		weights.append(layer.weights)
	data = np.array([weights, neuralnetwork.shape])

	if not overwrite and os.path.isfile(fileName + ".npy"):
		number = 1
		while os.path.isfile(makeNewName(fileName, number)):
			number += 1
		fileName = makeNewName(fileName, number)

	if printResults:
		print 'neural network saved at:', fileName
	np.save(fileName, data)


def makeNewName(fileName, number):
	return fileName + '(' + str(number) + ')' + ".npy"

def load(fileName):
	data = np.load(fileName + ".npy")
	weights = data[0]

	layers = []
	for weight in weights:
		layers.append(Layer(0,0))
		layers[-1].weights = weight
	neuralnetwork = NeuralNetwork()
	neuralnetwork.layers = layers
	return neuralnetwork

class Layer(object):
	def __init__(self, input_size, output_size, fn=sigmoid, magnitude=0):
		self.weights = np.random.randn(input_size + 1, output_size) / 10**magnitude
		self.activation_function = fn
		self.shape = (input_size, output_size)

	def feedforward(self, inputs):
		self.inputs = np.append(inputs, 1)
		z = np.dot(self.inputs, self.weights)
		self.outputs, self.derivative = self.activation_function(z)
		return self.outputs

	def backpropagate(self, errors):
		self.errors = errors * self.derivative
		return np.dot(self.weights, self.errors)[:-1]

	def updateWeights(self, step_size=1):
		dweight = np.outer(self.inputs, self.errors)
		A = np.linalg.norm(dweight)
		B = np.linalg.norm(self.weights)
		self.weights -= step_size * dweight

class NeuralNetwork(object):
	def __init__(self, layer_sizes=[], magnitude=0):
		# If new attributes are ever added, ensure to update save() and load()
		self.layers = [Layer(x, y, magnitude=magnitude) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

	@property
	def shape(self):
		return tuple([l.shape[1] for l in self.layers])

	def evaluate(self, inputs):
		for layer in self.layers:
			inputs = layer.feedforward(inputs)
		return inputs

	def learn(self, inputs, targets, step_size=1, errfn=crossEntropyError):
		outputs = self.evaluate(inputs)

		self.layers[-1].errors = errfn(self.layers[-1], targets, isDerivative=True)
		errors = np.dot(self.layers[-1].weights, self.layers[-1].errors)[:-1]

		for layer in reversed(self.layers[:-1]):
			errors = layer.backpropagate(errors)

		for layer in self.layers:
			layer.updateWeights(step_size)