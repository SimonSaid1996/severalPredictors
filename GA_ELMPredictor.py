import numpy as np
from myELM import myELM
'''
this class will create a GA_ELM machine(which is a simplified version of MEABP machine), this class will take in input_train, output_train, using Genetic Algorithm to encode
those inputs and outputs(also using the normal ELM machine or SVM to get the expected values), and then generating the best fit(in theory) inputs and outputs. after getting the 
best fit inputs and outputs, we use a back propagation process to get the weight and threash hold inside the ELM machine and use them as the starting weight and threash holds,
putting the input_train and the output_train into the refined version of the ELM machine again and the the better prediction
'''
class GA_ELMPredictor(object):
    def __init__(self,inputS, outputS, neuronS):
        '''
        this constructor will first create an ELM machine by inputS, outputS and neuronS. then this constructor will store the predicted value
        :param inputS:
                 input_train value of some datasets
        :param outputS:
                 output_train value of some datasets
        :param neuronS:
                 the neuron number of the assigned ELM machine
        '''
        #myELM(x.shape[1], 1,100)
        initialElm = myELM(inputS.shape[1], 1, neuronS)   #reshaping the matrix as needed, just the same as ELM predictor
        initialElm.train(inputS, outputS.reshape(-1, 1))
        output_predict = initialElm.predictResult(inputS)    #getting output predictions for all inputS and store them

        #to store all the input_train, output_train and output_predict values
        self.outputPredict = output_predict
        self.inputTrain = inputS
        self.outputTrain = outputS
        self.neuronSize = neuronS    #recording the neuronsize in advance

    def grouppingFunction(self, aValue):
        '''
        this function will generate a grouppingRange value which will help divide all the input_train data into groups
        :param aValue:
               a random generated number ranging from (0,1)
        :return:grouppingRange:
               grouppingRange, will help divide the total input_train data into several groups, ranging from (-20,20)
        '''
        grouppingRange = 40 * aValue - 20
        return grouppingRange

    def getWinNTempGroupFunction(self,curSubGroupScores,  groupCenterIndex):   #should have the outputvalue, outputpredict and outputscores subgroups, we divide outputvalue and outputpredict
        '''
        this function will input a subGroup and one groupCenter(the output with the maximum scoring function value), and output a win_group and a temp_group for later operations
        :param curSubGroupScores:
               the scores of the outputs within the subgroup, used to group outputs into win_group or temp_group
        :param groupCenterIndex:
               to help locate the groupCenter value inside of the subGroupOutputs and subGroupPredictions
        :return:win_group_index, temp_group_index
               to store the index of the win_group and temp_group outputs and the predictions(outputs and predictions use the same index in corresponding lists)
        '''
        win_group_index =[]; temp_group_index = []; #to store the index of the win_group and temp_group, later used to reference in subGroupOutputs and subGroupPredictions
        aValue = self.RandomValueGenerator()   #a is the random value generated ranging from (0,1)
        groupCenter = curSubGroupScores[ groupCenterIndex ]  #getting the groupCenter by the groupcenterindex and the curSubGroupScores
        dividingRange = groupCenter + 10 *( 2*aValue - 1 )   #a self defined function G(x)= ni+10*(2a-1) to help divide the subgroup, G(x) will range from (ni-10, ni+10), where ni represent the groupCenter
        #'''
        if dividingRange == groupCenter:  # situation where avalue ==0.5, only groupCenter can be divided into the win_group_index, other outputs all go into temp_group_index
            win_group_index.append(groupCenterIndex)
        for i in range( len( curSubGroupScores ) ):
            if dividingRange == groupCenter:  #situation where avalue ==0.5, all other outputs go into temp_group_index
                temp_group_index.append( i )
            elif dividingRange > groupCenter:   #situation where dividingRange > groupCenter, adding the values within [groupCenter,dividingRange] into the win_group_index and the rest go into temp_group_index
                if curSubGroupScores[ i ] >= groupCenter and curSubGroupScores[ i ] <= dividingRange:
                    win_group_index.append( i )
                else:
                    temp_group_index.append(i)
            else:   #situation where dividingRange < groupCenter, adding the values within [dividingRange, groupCenter] into the win_group_index and the rest go into temp_group_index
                if curSubGroupScores[ i ] <= groupCenter and curSubGroupScores[ i ] >= dividingRange:
                    win_group_index.append( i )
                else:
                    temp_group_index.append(i)
        #'''
        return win_group_index, temp_group_index

    def scoringFunction(self,output_train,output_predict):    #, scoringWithinGroup #only for testing
        '''
        this function will take the output_train matrix and the output_predict matrix, putting them into a ratting function and return a score,
        could also be used to judge the difference between different groups
        :param output_train:
                the real value of the training data
        :param output_predict:
                the prediction value of the training data
        :return score:
                the score of the differences to judge how much differences between output_train and output_predict
        '''
        score = 0 #default setting score as 0
        squareDifference = np.square(output_train - output_predict)
        score = 1 / squareDifference
        return score

    def RandomValueGenerator(self):
        '''
        this function will automatically generate a random aValue, ranging from 0-1.for the use of MFunction and the GFunction
        :return:
             aValue, a random value between (0,1)
        '''
        minValue = 0; maxValue = 1
        aValue = np.random.uniform(minValue, maxValue,1)
        return aValue

    def SelectWinningNTempSubgroups(self, originalOutputSubGroup, originalPredictSubgroup):
        '''
        this function will divide the originalSubGroup into 2 same sized lists: WinningSubGroupList and tempSubGroupList according to the grouppingValue
        :param originalSubGroup:
                the original subgroup list containing lists of output_trains
        :return: winningSubGroupIndexList, tempSubGroupIndexList
                winningSubGroupIndexList is the list with the index of outputs that are close to the subGroupBestOutput, tempSubGroupList is the list with the index of outputs that are far from the subGroupBestOutput
        '''
        winningSubGroupIndexList =[]; tempSubGroupIndexList =[]   #index groups for storing the wininingsubGroup and the tempSubgroup
        for i in range( len( originalOutputSubGroup ) ):
            curOutputGroup = originalOutputSubGroup[ i ]
            curPredictionGroup = originalPredictSubgroup[ i ]
            win_group_index=[]; temp_group_index=[]   #initializing the empty lists of index
            if len( curOutputGroup ) != 0:     #can only find the winningSubGroupList and tempSubGroupList if the curOutputGroup and the corresponding originalPredictionGroup are not null
                maxScore = -1; maxScoreIndex = -1  # to find out the max score by scoringFunction and record the index later, default setting those two values as -1
                curSubGroupScores = []
                for j in range( len( curOutputGroup )):
                    score = self.scoringFunction(curOutputGroup[ j ],curPredictionGroup[ j ])  #do this later   #only for testing, scoringWithinGroup= True
                    curSubGroupScores.append( score )        #store the scores, later used to divide the win_subgroup and the temp_subgroup
                    if score > maxScore:       #if finding greater scores than maxScore, update the maxScore and the corresponding index
                        maxScore = score
                        maxScoreIndex = j
                win_group_index, temp_group_index = self.getWinNTempGroupFunction( curSubGroupScores, maxScoreIndex)   #getting the win_group_index and temp_group_index,later used for crossover and mutate
            #if len(curOutputGroup) == 0, do nothing and directly adding the empty lists to get the same size of the wining/temp subgroupIndexList as the originaloutputSubgroups/originalpredictiongsubgroups
            winningSubGroupIndexList.append(win_group_index)
            tempSubGroupIndexList.append(temp_group_index)
        return winningSubGroupIndexList , tempSubGroupIndexList

    def getOutputMinNMax(self):
        '''
        this function will look at the output_train and get the minimum and the maximum value out of the output_train
        :return: outputMin, outputMax
                the min value and the max value from the output_train, to help dividing the output_train into N groups
        '''
        outputMin = min(self.outputTrain)
        outputMax = max(self.outputTrain)
        return outputMin, outputMax

    def getOriginalSubGroups(self, MFunctionResult, outputMin, outputMax):    #when MFunctionResult is negative here, there r issues, think about why
        '''
        this function will help  dividing the original output results into n subgroups, n is result of total length of the output_train divided by MFfunction
        :param MFunctionResult:
                    to help decide the subGroup range and the min and max of the values in the subgroups
        :param outputMin:
                    the min of the results of outputs, will become the start element of the subgroups
        :param outputMax:
                     the max of the result of outputs, will become the end element of the subgroups
        :return:originalSubgroups:
                     the originally divide subgroups, according to different range values
        '''
        subGroupRange = abs(MFunctionResult)  #use the MFunctionResult to get the subGroupRange for dividing subgroups
        subGroupNum = len(self.outputTrain)/subGroupRange
        isSubgroupMin = False                # an indicator to decide whether the element in the list is the max or the min of the subgroup, defaultly set as is not subGroupMin
        if MFunctionResult > 0:
            isSubgroupMin = True
        if isinstance(subGroupNum, int) is False:   #situation where the subgroupNum is not an integer ,need to add one more subgroup to add the remaining results
            subGroupNum = int(len(self.outputTrain)//subGroupRange +1)   #the remaining elements goes to the extra group
        diffSubGroupIndicator = outputMin   #first setting the subGroupIndicator starting from outputMin
        subGroupRangeList =[]    #adding the first element into the rangeList
        while diffSubGroupIndicator < outputMax:
            subGroupRangeList.append(diffSubGroupIndicator)                  #adding up the indicators into a list and process the data into another list later
            diffSubGroupIndicator = diffSubGroupIndicator + subGroupRange
        originalOutputSubgroups = [[] for i in range(subGroupNum)]   #settting up lists of empty lists, later adding output resutls into the originalSubgroups
        originalPredictionSubgroups = [[] for i in range(subGroupNum)]   #to store the corresponding outputpredictions
        originalInputSubgroups = [[] for i in range(subGroupNum)]        #to store the corresponding inputSubgroups
        outputPrediction = self.outputPredict.tolist()                   #converting the original matrix back to the list
        #'''
        
        for i in range( len( self.outputTrain ) ):
            output = self.outputTrain[ i ]
            subGroupListIndex = 0
            while subGroupListIndex < (len(subGroupRangeList) - 1):
                # iterating through the elements inside of the subGroupRangeList
                if isSubgroupMin:  # situation where the cur element stored in subGroupRangeList is min, MFunctionResult is positive
                    curRangeMin = subGroupRangeList[subGroupListIndex]
                    curRangeMax = subGroupRangeList[subGroupListIndex] + MFunctionResult
                else:  # situation where the cur element stored in subGroupRangeList is max, MFunctionResult is negative
                    curRangeMin = subGroupRangeList[subGroupListIndex] + MFunctionResult
                    curRangeMax = subGroupRangeList[subGroupListIndex]
                # then comparing with those two, if within the range, then add into the corresponding list, otherwise go to the next subGroupRangeList
                if output >= curRangeMin and output < curRangeMax:
                    # within cur subGroupRange list, store the result in the correspondingList
                    originalOutputSubgroups[ subGroupListIndex ].append( output )    #putting the output inside the list
                    originalPredictionSubgroups [ subGroupListIndex ].append( outputPrediction[ i ][ 0 ] )    #outputPredictions has the same corresponding index as output_train list
                    originalInputSubgroups [ subGroupListIndex ].append( self.inputTrain[ i ] )               #adding the inputs inside of the list
                    break
                else:  # outside the range, go to check the next subGroupRange
                    subGroupListIndex = subGroupListIndex + 1
        #'''
        return originalOutputSubgroups, originalPredictionSubgroups,originalInputSubgroups

    def select(self):
        '''
        this function will will generate N groups of input_train values, for the later genetic algorithm use
        :return:originalSubgroups:
               the original divided subgroups, later used for crossOver and Mutation
        '''
        #need to use the M function
        aValue = self.RandomValueGenerator()
        MFunctionResult = self.grouppingFunction( aValue )   #MFunction, will generate a MFunctionResult to help dividinig all values in input_train into N groups
        outputMin, outputMax = self.getOutputMinNMax()
        grouppingRange = MFunctionResult[0]       # the grouppingFunction always return a list of a float element, here to convert it back to the float
        originalOutputSubgroups, originalPredictionSubgroups, originalInputSubgroups = self.getOriginalSubGroups(grouppingRange, outputMin,outputMax)  #getting the original divided subgroups

        return originalOutputSubgroups, originalPredictionSubgroups,originalInputSubgroups

    def crossOver(self,originalOutputSubgroups, originalPredictionSubgroups):
        '''
        this function will input an originalOutputSubgroups and an originalPredictionSubgroups to evaluate all the outputs by scoringFunction, after that, generating a winingSubGroupList
        and a tempSubGroupList(represented by index, can de-reference by the index and the originalOutputSubgroups and originalPredictionSubgroups to get the exact values).
        :param originalOutputSubgroups:
                    subgroups of output values of the original trainingsets, to be input into the scoringFunction
        :param originalPredictionSubgroups:
                    subgroups of prediction values of the original trainingsets, to be input into the scoringFunction
        :return: winningSubGroupIndexList, tempSubGroupIndexList
                    the list of indexes of the winingSubGroup of the trainingsets and the list of indexes of the tempSubGroup of the trainningsets
        '''
        winningSubGroupIndexList, tempSubGroupIndexList = self.SelectWinningNTempSubgroups( originalOutputSubgroups, originalPredictionSubgroups)
        return winningSubGroupIndexList, tempSubGroupIndexList

    def ScoringFunctionSum(self, IndexList, outputList, predictionList):
        '''
        this function will help calculating the sum of scores in either the winningGroups or the tempGroups, just need into the IndexList(to help locate the value location within the outputList
        and the predictionList)
        :param IndexList:
                  list of index of the values within the outputList and predictionList
        :param outputList:
                  list of all output within the current subgroup
        :param predictionList:
                  list of all predictions within the current subgroup
        :return:listScore
                  the integer score of the subgroups, could be 0 if the size of the IndexList is 0(meanining no values within either the wininigSubGroup or the tempSubGroup)
        '''
        listLen = len( IndexList )
        listDenominatorSum = 0           #later used for the sum calculation
        if listLen != 0:                 #situation where there are values in the list
            for j in range(len(IndexList)):  # adding up the sum of the scoringFunction and compare, two subGroupIndexes sizes might not be the same
                curScore = self.scoringFunction(outputList[IndexList[j]],predictionList[IndexList[j]])
                listDenominatorSum = listDenominatorSum + curScore
            listScore = listLen/listDenominatorSum   #according to the function p(x)=n/sum((ti-yi)^2)
        else:                            #empty list, can't find predictions or outputs in the list, defaultly return listScore as 0
            listScore = 0
        return listScore

    def getBestFitInWinGroup(self, winGroupIndexList, curOutputSubgroup, curPredictionSubgroup, curInputSubgroup):
        '''
        this function will help get the best fit input and output from the current subgroup, will need to input the winGroupIndexList to locate the wingroup input,
        output and predictions
        :param winGroupIndexList:
                    a list index which indicates the index of the winining groups inside of the cur subgroups
        :param curOutputSubgroup:
                    current output subgroups, a fraction of the originalOutputSubgroup
        :param curPredictionSubgroup:
                    current prediction subgroups, a fraction of the originalPredictionSubgroup
        :param curInputSubgroup:
                    current input subgroups, a fraction of the originalInputSubgroup
        :return:bestFitInput,bestFitOutput
                    input and output with the highest score(using scoringFunction) inside of the subgroup
        '''
        bestFitInput = None; bestFitOutput = None    #default setting of the bestFit data, setting them to null at first
        maxScore = -1; maxScoreIndex = -1   #defaultly setting the maxScoreIndex as -1, the maxScore as -1
        for i in range( len( winGroupIndexList ) ):
            score = self.scoringFunction( curOutputSubgroup[ winGroupIndexList[ i ] ], curPredictionSubgroup[ winGroupIndexList[ i ] ] )   #putting the winGroupIndex inside the outputSubgroup and predictionSubgroup to get score
            if score > maxScore:  # if finding greater scores than maxScore, update the maxScore and the corresponding index
                maxScore = score
                maxScoreIndex = winGroupIndexList[ i ]
        #finding the bestfitInput and bestFitOutput by maxScoreIndex
        bestFitInput = curInputSubgroup[maxScoreIndex]
        bestFitOutput = curOutputSubgroup[maxScoreIndex]
        return bestFitInput, bestFitOutput

    def mutation(self,winningSubGroupIndexList, tempSubGroupIndexList,originalOutputSubgroups, originalPredictionSubgroups, originalInputSubgroups):
        '''
        this function will input winningSubGroupIndexList, tempSubGroupIndexList,originalOutputSubgroups, originalPredictionSubgroups and use the scoringFunction to get the
        sum of scores in the wining and temp groups, if meeting the requirement, will swap the wining and temp groups and in the end, output a lists of results in which each result
        has one exact best-fit training entity for later calculation of the best starting parameter and threashhold of a new ELM machine
        :param winningSubGroupIndexList:
                         indexlist to store the winingSubgroup of training data
        :param tempSubGroupIndexList:
                         indexlist to store the tempSubgroup of training data
        :param originalOutputSubgroups:
                         real outputs of the trainning data
        :param originalPredictionSubgroups:
                         predicting values of the training data
        :return:bestFitTrainingDataInputs, bestFitTrainingDataOutputs
                         list of best fit training data inputs and outputs for later generating the best starting parameters and the threashholds
        '''
        #need to use the winingsubgroupindexlist and the tempindexlist to calculate the scoringfunction of the entire 2 subgroups, swapping them if necessary and get the best fit list of training data
        bestFitTrainingDataInputs =[]; bestFitTrainingDataOutputs =[]  #to store the list of bestFitTrainingDatas
        for i in range( len( originalOutputSubgroups )):    #all four lists has the same data, iterating through one of them to use the index
            if len( originalOutputSubgroups[ i ] ) != 0:    #situations where wininngsubsets and tempSubsets are not all empty
                curWiningSubGroupIndexes = winningSubGroupIndexList[ i ]
                curTempSubGroupIndexes = tempSubGroupIndexList[ i ]
                winGroupScoreSum = 0; tempGroupScoreSum = 0            #the sums of two groups, later used to compare the sum and swap

                #calculate the winGroupScoreSum and tempGroupScoreSum and compare the value of them, if tempGroupScoreSum value is greater than winGroupScoreSum, then do a swap
                winGroupScoreSum = self.ScoringFunctionSum( curWiningSubGroupIndexes, originalOutputSubgroups[ i ], originalPredictionSubgroups[i])
                tempGroupScoreSum = self.ScoringFunctionSum( curTempSubGroupIndexes, originalOutputSubgroups[ i ], originalPredictionSubgroups[i])

                if tempGroupScoreSum > winGroupScoreSum:     #situation in which tempSubGroup and winingSubGroup need to swap location
                    swappingIndexlist = curWiningSubGroupIndexes
                    winningSubGroupIndexList[ i ] = curTempSubGroupIndexes    #swapping the two index groups if tempGroupScoreSum > winGroupScoreSum
                    tempSubGroupIndexList[ i ] = swappingIndexlist
                    #then do another time of getting the best fit data for a subGroup each time and then output the list result, using winingSubGroupIndexList to get the best fit
                    curBestFitInput, curBestFitOutput = self.getBestFitInWinGroup( winningSubGroupIndexList[ i ], originalOutputSubgroups[ i ], originalPredictionSubgroups[i], originalInputSubgroups[i])
                    bestFitTrainingDataInputs.append( curBestFitInput )
                    bestFitTrainingDataOutputs.append( curBestFitOutput )

        return bestFitTrainingDataInputs, bestFitTrainingDataOutputs

    def backMappingFunction(self,maxOutput,minOutput,outputList):
        '''
        this function will backMapping the bestFitTrainingDataOutputs into a list ranged in ( 0,1 ), later be used to represent the output of the ELM machine,
        because the sigmoid function result will increase as the input increases, so this function will mimic the location after the sigmoid function
        :param maxOutput:
                  the maxOutput of the outputList, used to help setting up the range
        :param outputList:
                  a list of the outputs, used as the domain values of the function
        :return: backMappingResultList
                  a list of outputs after backMapping,later will be used to find the original weight and threashhold values
        '''
        backMappingResultList = []    #the list to store all the backmapping results
        for output in outputList:
            backMappingResult = (output- minOutput+1)/(maxOutput-minOutput+2)      #to make sure that the backMappingResult ranging from (0,1)
            #backMappingResult = output/(1+maxOutput)   #assuming all the outputs are positive numbers
            backMappingResultList.append(backMappingResult)
        return backMappingResultList

    def sigmoidFunc( self, Processed_input ):     #need to try if other activation functions work, could try tanh function here
        '''
        this function will mimic the use of the sigmoid function, inputing the processed_input and output the sigmoid result
        :param processed_input:
               processed_input(input after times the weight and adding up the threashholds)
        :return: the result after sigmoid function
        '''
        return 1 / (1 + np.exp(-Processed_input) )#sigmoid

    def findStartWeightNThreash(self,inputsets, outputsets):
        '''
        this function will first initialize a random value weight and a random value threash, then use Gradient Descent method, keep inputing the bestfit values to
        simulate the process of finding the best starting weight and threash values
        :param inputset:
                 the best fit inputset of the training data
        :param outputset:
                 the best fit outputset of the trainnig data
        :return:startingWeight, startingThreash
                 the starting value for weight and threashhold values, later used in another ELM machine to help predict the output values
        '''
        lowerEnd = -0.5; higherEnd = 0.5  # starting with random range, within (-1,1) will be fine
        weight = np.matrix(np.random.uniform(lowerEnd, higherEnd, (self.neuronSize, len(inputsets[0] ))))  # input weight matrix has to relate to the neuron and inputset size
        outputLowerEnd = 0; outputHigherEnd = 1; rowNum = 1  # sigmoid function's output range as the range here
        threash = np.matrix(np.random.uniform(outputLowerEnd, outputHigherEnd, (rowNum, self.neuronSize)))

        step_Len = 0.01  #as changing persent of the weight, the smaller the better but within a limit
        #only need the outputlayer input and output, and things up to g to calculate will be fine
        for i in range( len(inputsets) ):
            curInputset = np.mat( inputsets[ i ] ).astype(np.float64)      #getting the curinputset, change into the matrix form
            curOutputset = np.mat( outputsets[ i ] ).astype(np.float64)    #getting the curoutputset, change into the matrix form

            # input*weight
            weightedInput = np.dot(curInputset, weight.T).astype(np.float64)
            # proceed to minus the threash and use the sigmoid function
            sigmoidResult = self.sigmoidFunc(weightedInput - threash).astype(np.float64)

            a = np.multiply(sigmoidResult, 1 - sigmoidResult)  # part of the formula of slop changes in threash
            g = np.multiply(a, curOutputset - sigmoidResult)  # the changging value of threashhold

            threashHold_change = -step_Len * g  #the change of threashhold
            weight_change = step_Len * np.dot(np.transpose(curOutputset), g).T  #the change of weight, remember to do transpose to fit the dot product

            threash += threashHold_change
            weight += weight_change

        return weight, threash    #startingWeight, startingThreash
