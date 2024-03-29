import math
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
        for node in self.output: #output layer errors
            node.errorRate = node.expectedOutput[outputindex] - node.outputs
            error.append(node.errorRate)
            #print(node.nodeNum,"--->",node.errorRate)
        for node in self.hidden:#hidden layer errors 
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
    
    #probability distribution
    def softmax(self):
        den = 0.0
        probability_distribution = []
        print("Outputs:")
        for node in self.hidden + self.output:
            print(node.nodeNum,"--->",node.outputs)
        for node in self.output:
            den += math.exp(node.outputs)
        for node in self.output:
            Numerator = (math.exp(node.outputs) / den)
            probability_distribution.append(Numerator)
        print("Softmax: \n",probability_distribution)



            
        
def graph(Epoch , SQ_error):
    x = Epoch
    y = SQ_error
    
    plt.plot(x,y)
    # naming the x axis
    plt.xlabel('x - Epoch')
    # naming the y axis
    plt.ylabel('y - Squared error')
    # giving a title to my graph
    plt.title('Learning Curve')
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

#input value for number of epochs 
while True:
    try:
        User_input = int(input("Number of Epochs: "))
        break
    except ValueError:
        print("invalid input")
        
Epoch = []
RealError = []
TableWeights = []
for epoch in range(User_input):
    SQ_error = []
    Epoch.append(epoch+1)
    print("\n","epoch ------> ", epoch + 1 )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for j in data:
        index = data.index(j)
        for node in nodelist:
            node.inputs.clear()
            for i in j:
                node.inputs.append(i)#adds the new line of data 
        n.forwardstep()
        n.error(index)
        n.weightUpdate()
        SQ_error.append(n.squared_error())
    avg_error = 0
    for error in SQ_error:
        avg_error += error
    #avg_error = avg_error / 6 
    RealError.append(avg_error)
    
# selecting whether they want to test the data 
while True:
    try:
        input1 = input("Would you like to test your data? Y/N:").lower()#converts input into lower case 
        if input1 == "y":
            for node in nodelist:
                node.inputs.clear()
                node.inputs = [1,0.3,0.7,0.9]
            n.forwardstep()
            n.softmax()
            break
        elif input1 == "n":
            pass
            break
    except ValueError:
        print("invalid input")



graph(Epoch, RealError)