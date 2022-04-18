import math
import random
import matplotlib.pyplot as plt

class node:
    
    def __init__(self,_nodeNum, _weights, _inputs, _outputs,_expectedOutput,_errorRate):
        self.inputs = _inputs
        self.outputs = _outputs 
        self.weights = _weights
        self.nodeNum = _nodeNum
        self.expectedOutput = _expectedOutput
        self.errorRate = _errorRate

        
    def net(self):
        net_sum = 0.0
        for i in range(0,len(self.inputs)):
            net_sum += (self.weights[i] * self.inputs[i])
        return net_sum
    
    def sigmoid(self):
       return  1 / (1 + math.exp(-self.net()))
            
   
   
   
class Network():
    def __init__(self,_hidden,_output):
        self.hidden = _hidden
        self.output = _output
    
    def forwardstep(self):
        fwoutput = []
        for node in self.hidden:
            fwoutput.append(node.sigmoid())
            node.outputs = node.sigmoid()
        fwoutput = [1] + fwoutput
        for node in self.output:
            node.inputs = fwoutput
        fwoutput = []
        for node in self.output:
            fwoutput.append(node.net())
            node.outputs = node.net()
        #for node in self.hidden + self.output:
            #print(node.nodeNum,"--->",node.outputs)
    
    def error(self,outputindex):
        error = []
        for node in self.output: 
            node.errorRate = node.expectedOutput[outputindex] - node.outputs
            error.append(node.errorRate)
            #print(node.nodeNum,"--->",node.errorRate)
        for node in self.hidden:
            W = []
            for outputnode in self.output:
                W.append(outputnode.weights[self.hidden.index(node) + 1])
            node.errorRate = (node.outputs * (1 - node.outputs) * ((error[0] * W[0] + error[1] * W[1])))

    def weightUpdate(self):
        for node in self.hidden:
            for i in range(len(node.inputs)):
               node.weights[i] += 0.1 * node.errorRate * node.inputs[i] 
        for node in self.output:
            for i in range(len(node.inputs)):
               node.weights[i] += 0.1 * node.errorRate * node.inputs[i] 
        for node in self.hidden + self.output:
            print(node.nodeNum,"--->",node.weights,"\n")

    def squared_error(self):
        sumn = 0
        for node in self.output:
            sumn += (node.errorRate ** 2)

        return sumn * 0.5
        
def graph(Epoch , SQ_error):
    x = Epoch
    y = SQ_error
    
    plt.plot(x,y)
    # naming the x axis
    plt.xlabel('x - Epoch')
    # naming the y axis
    plt.ylabel('y - Squared error')
    # giving a title to my graph
    plt.title('squared error graph')
    plt.show()
        

#list of nodes 
nodelist = []
node4 = node(4,[0.9,0.74,0.8,0.35],[],[],[],0)      
nodelist.append(node4)       
node5 = node(5,[0.45,0.13,0.4,0.97],[],[],[],0)
nodelist.append(node5)
node6 = node(6,[0.36,0.68,0.1,0.96],[],[],[],0)
nodelist.append(node6)
node7 = node(7,[0.98,0.35,0.5,0.9],[],[],[1,1,1,0,0,0],0)
nodelist.append(node7)
node8 = node(8,[0.92,0.8,0.13,0.8],[],[],[0,0,0,1,1,1],0)
nodelist.append(node8)
    

data = [[1,0.50,1.00,0.75],
[1,1.0,0.50,0.75],
[1,1.00,1.00,1.00],
[1,-0.01,0.50,0.25],
[1,0.50,-0.25,0.13],
[1,0.01,0.02,0.05]]

#seperating the nodes into the specific layers 
hiddenLayer = [node4,node5,node6]
outputLayer = [node7,node8]

#MAIN CODE
n = Network(hiddenLayer,outputLayer)


while True:
    try:
        p = int(input("Number of Epochs: "))
        break
    except ValueError:
        print("invalid input")
        
Epoch = []
RealError = []
for epoch in range(p):
    SQ_error = []
    Epoch.append(epoch+1)
    print("\n","epoch ------> ", epoch + 1 )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for j in data:
        index = data.index(j)
        for node in nodelist:
            node.inputs.clear()
            for i in j:
                node.inputs.append(i)
        n.forwardstep()
        n.error(index)
        n.weightUpdate()
        SQ_error.append(n.squared_error())
    avg_error = 0
    for error in SQ_error:
        avg_error += error
    avg_error = avg_error / 6 
    RealError.append(avg_error)


graph(Epoch, RealError)