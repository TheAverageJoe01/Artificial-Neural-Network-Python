import math
import random
 
# Initialize a network
def Network_start(Inputs, n_hidden, Outputs):
	network = list()
	Hidden_L = [{'weights':[0.90,0.45,0.36,0.74,0.13,0.68,0.80,0.40,0.10,0.35,0.97,0.96]}]
	network.append(Hidden_L)
	Ouput_L = [{'weights':[0.98,0.92,0.35,0.80,0.50,0.13,0.90,0.80]}]
	network.append(Ouput_L)
	return network
 
# Calculate node net for an input
def Net(weights, input):
	net = weights[-1]
	for i in range(0,len(weights)):
		net += weights[i] * input[i]
	return net
 
# Sigmoid node net
def Sigmoid(net):
	return 1.0 / (1.0 + math.exp(-net))
 
# Foward step
def forward_step(network, row):
	input = row
	for layer in network:
		temp = []
		for node in layer:
			net = Net(node['weights'], input)
			node['output'] = Sigmoid(net)
			temp.append(node['output'])
		input = temp
	return input
 
# Calculate the  sigmoid derivative 
def Sigmoid_derivative(output):
	return output * (1.0 - output)
 
# Backward step
def backward_step(network, target):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for node in network[i + 1]:
					error += (node['weights'][j] * node['x'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				node = layer[j]
				errors.append(node['output'] - target[j])
		for j in range(len(layer)):
			node = layer[j]
			node['x'] = errors[j] * Sigmoid_derivative(node['output'])
 
# Update network weights with error
def update_weights(network, row, Learning_rate):
	for i in range(len(network)):
		input = row[:-1]
		if i != 0:
			input = [node['output'] for node in network[i - 1]]
		for node in network[i]:
			for j in range(len(input)):
				node['weights'][j] -= Learning_rate * node['x'] * input[j]
			node['weights'][-1] -= Learning_rate * node['x']
 
# Train a network for a fixed number of epochs
def train_network(network, train, Learning_rate, epoch, Outputs):
	for epoch in range(epoch):
		sum_error = 0
		for row in train:
			output = forward_step(network, row)
			target = [0 for i in range(Outputs)]
			target[row[-1]] = 1
			sum_error += sum([(target[i]-output[i])**2 for i in range(len(target))])
			backward_step(network, target)
			update_weights(network, row, Learning_rate)
		print(f"Epoch = {epoch + 1} \nError = {sum_error} \n~~~~~~~~~~~~" )

random.seed(1)

data = [[0.50,1.00,0.75,1],
    [1.0,0.50,0.75,1],
    [1.00,1.00,1.00,1],
    [-0.01,0.50,0.25,2],
    [0.50,-0.25,0.13,2],
    [0.01,0.02,0.05,2]]

Inputs = len(data[0]) - 1
Outputs = len(set([row[-1] for row in data]))
network = Network_start(Inputs, 3, Outputs)
print(network)
#train_network(network, data, 0.1, 10, Outputs)
#for layer in network:
#	print(layer)
