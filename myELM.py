import numpy as np
"""
this file will create an ELM(extreme learning machine) class which enalbe users to input some data and some output data and then create a module to predict later outputs
it will store the params inside some matrixes and then we can input new data and use the inverse matrices to get the predicted result 
"""

class myELM(object):
    def __init__(self,inputS, outputS, neuronS):
        '''
        initialize the elm class with user inputsize and the user outputsize,also initialize the machine's weight and threashold with random number
        will also initialize a weightMatrix and a pseudoInputMatrix
        :param inputS: int
               the number of input layer that users want
        :param outputS: int
               the number of output layer that users want
        :param neuronS: int
               the number of neurons which hide in the hidden layer
        '''
        self.inputSize = inputS
        self.outputSize = outputS
        self.neuronSize = neuronS

        lowerEnd = -0.5; higherEnd = 0.5   #just random number range, can be changed at any time
        self.weight = np.matrix(np.random.uniform(lowerEnd, higherEnd, (self.neuronSize, self.inputSize))) #input weight matrix has to relate to the neuron and inputsize
        outputLowerEnd = 0; outputHigherEnd = 1; rowNum =1   #sigmoid function's output range as the range here
        self.threash = np.matrix(np.random.uniform(outputLowerEnd, outputHigherEnd, (rowNum, self.neuronSize))) #threash matrix is related to the neuronsize and should only have one row

        self.pseudoInputMatrix = 0 #np.matrix(0)#0         #default setting pseudoInputMatrix and weightMatrix as 0
        self.weightMatrix = 0 #np.matrix(0)  #need to do the corresponding size as input

    def updateStartingWeightNThreash(self,newWeight, newThreash):
        '''
        this function will allow users to self-define the starting weight and starting threash, normally cooperate with GA_ELMPredictor
        :param newWeight:
                   the new starting weight the users can define
        :param newThreash:
                   the new starting threash the users can define
        '''
        self.weight = newWeight
        self.threash = newThreash

    def train(self,input, output):
        """
        :param input:
              an input matrix, containning input object features
        :param output:
              an output matrix, containning output result
        :return:
              the product of pseudoInputMatrix and weightMatrix as the result of the training process(containing inverse mapping product)
        """
        input = np.matrix( input )
        output= np.matrix( output )          #later need matrix calculation, so need to turn input and output into matrices
        self.pseudoInputMatrix = ( input * self.weight.T  )+ self.threash
        self.pseudoInputMatrix = self.sigmoidFunc( self.pseudoInputMatrix )
        #process the inputdata into y=ax+b and the sigmoid function

        moore_penrose_pseudoInput = np.linalg.inv(self.pseudoInputMatrix.T * self.pseudoInputMatrix)* self.pseudoInputMatrix.T

        self.weightMatrix = moore_penrose_pseudoInput*output
        #calculating according to the function H' = (Ht*H)^-1 * Ht, weightmatrix = H'* output

        return self.pseudoInputMatrix * self.weightMatrix    #convert the origianl matrix back

    def predictResult(self, input ):
        input = np.matrix( input )  #process input into a matrix
        output = self.sigmoidFunc( (input* self.weight.T)+self.threash )* self.weightMatrix
        #according to the calculating function to get the outputmatrix
        return output

    def sigmoidFunc( self, input ):     #need to try if other activation functions work, could try tanh function here
        return 1 / (1 + np.exp(-input) )#sigmoid
        # #tanh (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        #np.where(input < 0, 0.5 * input, input)  #perlu, relu can't be directly applied here, need to change the model entirely
