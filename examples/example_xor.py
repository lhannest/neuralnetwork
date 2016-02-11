from src import neuralnetwork

def truthTable (n):
# Credit: http://stackoverflow.com/a/6336676
	if n < 1:
		return [[]]
	subtable = truthTable(n-1)
	return [ row + [v] for row in subtable for v in [0,1] ]

def xor(*args):
	truth_count = 0
	for p in args:
		if p: truth_count += 1
	return truth_count == 1

inputs = truthTable(4)
targets = [[xor(a, b, c, d)] for a, b, c, d in inputs]

nnet = neuralnetwork.NeuralNetwork([4, 2, 1])

for i in range(1000):
	error = 0
	for x, t in zip(inputs, targets):
		error += nnet.learn(x, t)
	if i%10 == 0:
		print 'Iteration:', i, 'Error:', error

for x, t in zip(inputs, targets):
	y = nnet.evaluate(x)[0]
	print x, y, round(y) == t[0]