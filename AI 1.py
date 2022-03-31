import math
import random
 
class node():
    
    def __init__(self, _weights, _inputs, _outputs, _bias, _net_sum):
        self.inputs = _inputs
        self.outputs = _outputs 
        self.bias = _bias
        self.weights = _weights
        self.net_sum = _net_sum
    
    def sigmoid(self):
        self.outputs = (1/1 + math.exp(-self.net_sum))
        
    def net(self, inputs):
        self.net_sum = 0
        for i in range(0,len(inputs)):
            for j in range(0,len(self.weights)):
                self.net_sum += self.weights[i] * inputs[i]
                
    def forwardstep(nodelist, data):
        for i in range(0,len(nodelist)):
            if nodelist[i] == nodelist[3] or nodelist[i] == nodelist[4]:
                nodelist.net(data[i])
                print(nodelist[i].outputs)
            else:
                nodelist[i].net(data[i])
                nodelist[i].sigmoid()
            
            
    def backwardstep(nodelist, data):
        pass
        
        
             

nodelist = []
node1 = node([0.9,0.74,0.8,0.35],[],0,0,0)      
nodelist.append(node1)       
node2 = node([0.45,0.13,0.4,0.97],[],0,0,0)
nodelist.append(node2)
node3 = node([0.36,0.68,0.1,0.96],[],0,0,0)
nodelist.append(node3)
node4 = node([0.98,0.35,0.5,0.9],[],0,0,0)
nodelist.append(node4)
node5 = node([0.92,0.8,0.13,0.8],[],0,0,0)
nodelist.append(node5)
    

data = [[1,0.50,1.00,0.75],
    [1,1.0,0.50,0.75],
    [1,1.00,1.00,1.00],
    [1,-0.01,0.50,0.25],
    [1,0.50,-0.25,0.13],
    [1,0.01,0.02,0.05]]


node.forwardstep(nodelist, data)