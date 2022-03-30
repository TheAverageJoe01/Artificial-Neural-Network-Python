import math
import random
 
class node(weights, inputs , output , bias):
    
    def __init__(self, _inputs, _outputs, _bias, _weights, _net_sum):
        self.inputs = _inputs
        self.outputs = _outputs 
        self.bias = _inputs[0]
        self.weights = _weights
        self.net_sum = _net_sum
    
    def sigmoid(self):
        self.outputs = (1/1 + math.exp**(-self.net_sum))
        
    def net(self):
        self.net_sum = 0
        for i in range(0,len(self.inputs)):
            for j in range(0,len(self.weights)):
                self.net_sum += self.weights[i] * self.inputs[i]
                
    def forwardstep(self):
        


node1 = node([0.9,0.74,0.8,0.35])             
node2 = node([0.45,0.13,0.4,0.97])
node3 = node([0.36,0.68,0.1,0.96])
node4 = node([0.98,0.35,0.5,0.9])
node5 = node([0.92,0.8,0.13,0.8])
    

data = [[0.50,1.00,0.75,1],
    [1.0,0.50,0.75,1],
    [1.00,1.00,1.00,1],
    [-0.01,0.50,0.25,2],
    [0.50,-0.25,0.13,2],
    [0.01,0.02,0.05,2]]
