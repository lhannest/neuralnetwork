import numpy as np
import os.path
import pudb

def sigmoid(x):
	y = 1 / (1 + np.exp(-x))
	return y, y * (1 - y)

def getSquaredError(errors):
	squared_error = 0
	for error in errors:
		squared_error += error**2
	return squared_error / 2

def save(neuralnetwork, fileName, overwrite=False):
	weights = []
	for layer in neuralnetwork.layers:
		weights.append(layer.weights)
	data = np.array([weights, neuralnetwork.shape])

	if not overwrite and os.path.isfile(fileName + ".npy"):
		number = 1
		while os.path.isfile(makeNewName(fileName, number)):
			number += 1
		fileName = makeNewName(fileName, number), data)

	np.save(fileName + ".npy", data)

def makeNewName(fileName, number):
	return fileName + '(' + str(number) + ')' + ".npy"

def load(fileName):
	data = np.load(fileName + ".npy")
	weights = data[0]
	shape = data[1]

	layers = []
	for weight in weights:
		layers.append(Layer(0,0))
		layers[-1].weights = weight
	neuralnetwork = NeuralNetwork([])
	neuralnetwork.layers = layers
	neuralnetwork.shape = shape
	return neuralnetwork

class Layer(object):
	def __init__(self, input_size, output_size, f=sigmoid):
		self.weights = np.random.randn(input_size + 1, output_size)
		self.activation_function = f

	def feedforward(self, inputs):
		self.inputs = np.append(inputs, 1)
		z = np.dot(self.inputs, self.weights)
		outputs, self.derivative = self.activation_function(z)
		return outputs

	def learn(self, inputs, targets, step_size=1):
		outputs = self.feedforward(inputs)
		errors = (outputs - np.array(targets)) * self.derivative
		self.weights -= step_size * np.outer(self.inputs, errors)

	def backpropagate(self, errors):
		self.errors = errors * self.derivative
		return np.dot(self.weights, self.errors)[:-1]

	def updateWeights(self, step_size=1):
		self.weights -= step_size * np.outer(self.inputs, self.errors)

class NeuralNetwork(object):
	def __init__(self, layer_sizes):
		# If new attributes are ever added, ensure to update save() and load()
		self.layers = [Layer(x, y) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
		self.shape = tuple(layer_sizes)

	def evaluate(self, inputs):
		for layer in self.layers:
			inputs = layer.feedforward(inputs)
		return inputs

	def learn(self, inputs, targets, step_size=1):
		outputs = self.evaluate(inputs)
		errors = outputs - np.array(targets)
		squared_error = getSquaredError(errors)

		for layer in reversed(self.layers):
			errors = layer.backpropagate(errors)

		for layer in self.layers:
			layer.updateWeights(step_size)

		return squared_error