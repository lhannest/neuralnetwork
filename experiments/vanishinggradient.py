import numpy as np
from src import neuralnetwork
from src import printer
from data import mnist

path = '/home/lance/git/neuralnetwork/data'

def mux(number):
	result = np.zeros(10)
	result[number] = 1
	return result

def demux(numbers):
	highest_index = 0
	for i, number in enumerate(numbers):
		if number > numbers[highest_index]:
			highest_index = i
	return highest_index

# Images are 28x28 pixels. Reshapen, they are 784x1
training = [(mux(label), image.reshape(784)/255.0) for label, image in mnist.read(path=path, dataset="training")]
testing = [(mux(label), image.reshape(784)/255.0) for label, image in mnist.read(path=path, dataset="testing")]
nnet = neuralnetwork.NeuralNetwork([784, 40, 30, 20, 10])