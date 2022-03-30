import math
import random
 
 
class node(weights, inputs , output , bias):
    
    def __init__(self, _inputs, _outputs, _bias, _weights, _net_sum):
        self.inputs = _inputs
        self.outputs = _outputs 
        self.bias = _inputs[0]
        self.weights = _weights
        self.net_sum = _net_sum
    
    def net(self):
        self.net_sum = 0
        for i in range(0,len(self.inputs)):
            for j in range(0,len(self.weights)):
            	self.net_sum += self.weights[i] * self.inputs[i]
             
    def sigmoid(self):
                
        

        
    
        
        
    
    



data = [[0.50,1.00,0.75,1],
    [1.0,0.50,0.75,1],
    [1.00,1.00,1.00,1],
    [-0.01,0.50,0.25,2],
    [0.50,-0.25,0.13,2],
    [0.01,0.02,0.05,2]]
