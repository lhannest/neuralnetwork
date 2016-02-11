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
nnet = neuralnetwork.NeuralNetwork([784, 30, 10])

ITR=30
BTC=3
p = printer.printer(1)
for i in range(ITR):
	error = 0
	for t, x in training:
		for j in range(BTC):
			error += nnet.learn(x, t, 1)
		message = 'Iteration ' + str(i) +', error ' + str(error)
		p.overwrite(message)
p.clear()

success = 0
for t, x in training:
	y = demux(nnet.evaluate(x))
	if y == demux(t):
		success += 1

print 'On training data ' + str(success*100.0 / len(training)) + '% success'

success = 0
for t, x in testing:
	y = demux(nnet.evaluate(x))
	if y == demux(t):
		success += 1

print 'On testing data ' + str(success*100.0 / len(testing)) + '% success'

neuralnetwork.save(nnet, 'saves/mnist_net')