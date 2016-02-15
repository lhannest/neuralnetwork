from src import neuralnetwork
from src import printer
from data import mnist
import numpy as np
import pudb


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

def learn(self, inputs, targets, step_size=1):
	outputs = self.evaluate(inputs)
	errors = outputs - np.array(targets)

	for layer in reversed(self.layers):
		errors = layer.backpropagate(errors)

	previous_dir = None
	for layer in reversed(self.layers):
		previous_dir = updateWeights(layer, step_size, previous_dir)

def updateWeights(self, step_size=1, previous_dir=None):
	# pudb.set_trace()
	dx_W = np.outer(self.inputs, self.errors)
	if previous_dir == None:
		previous_dir = dx_W
	ratio = np.mean(previous_dir) / np.mean(dx_W)
	self.weights -= step_size * dx_W * ratio
	return dx_W

# Images are 28x28 pixels. Reshapen, they are 784x1
training = [(mux(label), image.reshape(784)/255.0) for label, image in mnist.read(path=path, dataset="training")][:6000]
testing = [(mux(label), image.reshape(784)/255.0) for label, image in mnist.read(path=path, dataset="testing")][:1000]
nnet = neuralnetwork.NeuralNetwork([784, 30, 30, 30, 10])

ITR = 3
p = printer.Printer(0.5)
iteration = 0
total = ITR * len(training)
for i in range(ITR):
	for t, x in training:
		iteration += 1
		learn(nnet, x, t, 1)
		per = iteration * 100 / total
		p.overwrite('Learning: ' + str(per) + '%')
p.clear()

success = 0
for t, x in testing:
	if demux(nnet.evaluate(x)) == demux(t):
		success += 1

accuracy = int(success * 100 / len(testing))
print 'Accuracy: ' + str(accuracy) + '%'

# 53.12
# 51.16