import sys
path = '/home/lance/git/neuralnetwork/data'
sys.path.append(path)
import mnist
import numpy as np
import neuralnetwork
import printer
import pudb

def mux(number):
	# pudb.set_trace()
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
training = [(mux(label), image.reshape(784)/255.0) for label, image in mnist.read(path=path, dataset="training")][:100]
testing = [(mux(label), image.reshape(784)/255.0) for label, image in mnist.read(path=path, dataset="testing")][:100]
nnet = neuralnetwork.NeuralNetwork([784, 30, 10])

ITR=30
BTC=3
p = printer.printer(1)
for i in range(ITR):
	error = 0
	for t, x in training:
		for j in range(BTC):
			error += nnet.learn(x, t, 3)
		message = 'Iteration ' + str(i) +', error ' + str(error)
		p.overwrite(message)
p.clear()

success = 0
for t, x in testing:
	y = demux(nnet.evaluate(x))
	if y == demux(t):
		success += 1

print 'Finished with a success of ' + str(success*100.0 / len(testing))