import numpy as np
from sklearn import  datasets   #to get testing datasets
from sklearn.model_selection import train_test_split
from myELM import myELM
from gridSearchSVM import gridSearchSvm
from GA_ELMPredictor import GA_ELMPredictor

'''author ziran cao, a simple extrem learning machine to get user input, and output predicting results
this program will use the myELM class to input and predict some results, has referenced the website:
https://baijiahao.baidu.com/s?id=1654059102929159764&wfr=spider&for=pc

'''
def getCorrectionRate(predicting_data,actual_data):
    totalErrorRate = 0    #to set totalErrorRate
    for i in range(len(predicting_data)):
        predictElement = predicting_data[i]
        actualElement = actual_data[i]
        errorRate = abs(actualElement - predictElement)/actualElement     #get the absolute difference and find out the error rate
        totalErrorRate += errorRate
    totalErrorRate = totalErrorRate / len(predicting_data)      #sum of the total errors divided by the num of the errors to get the average
    #print(errorRate)
    return errorRate

def getCorrectionRate1(predicting_data,actual_data):
    print('below are the errorRate')
    bigErrorRateCount = 0
    for i in range(len(predicting_data)):
        predictElement = predicting_data[i]
        actualElement = actual_data[i]
        errorRate = (actualElement - predictElement)/actualElement #abs(yElement - xElement)/yElement     #get the absolute difference and find out the error rate
        if errorRate >=1 or errorRate <= -1:
            bigErrorRateCount = bigErrorRateCount +1
        print(errorRate)
    print('bigErrorRate is ',bigErrorRateCount)

def process_and_predict(input_train, input_test, output_train,output_test):
    testingElm = None  # first set up the elm machine as None and then do the update
    breakIndicator = False
    # then train the data
    errorRate = 1
    targetErrorRate = 0.15  # have to hit the correction rate up to 90%, 0.2 might be a better and faster num
    numCount = 0
    while errorRate > targetErrorRate:
        testingElm = myELM(input_train.shape[1], 1,
                           100)  # reshape the input into a matrix of (500,1), output is 1 and neuronsize is 100
        testingElm.train(input_train, output_train.reshape(-1, 1))  # train data
        output_train_predict = testingElm.predictResult(input_train)
        errorRate = getCorrectionRate(output_train_predict, output_train)
        numCount = numCount + 1
        if numCount == 1000:  # in case of infinite looping
            breakIndicator = True       #invalid result, shouldn't be counted to the final result
            print(
                'no errorRate has met the targetErrorRate requirement, please restart the program, too many times of looping')
            break
    print('cur error rates is ',
          errorRate)  # error rate still pretty fluctuating, could go from  0.01-3, most times r around 0.3-0.5

    output_predict = testingElm.predictResult(input_test)
    getCorrectionRate1(output_predict, output_test)

    return output_predict, breakIndicator

def getMidResult( predictingList, output_test ):    #need to compare the difference between different result, and get the medium of the resultList
    #every terms starts from the output_test, each terms need to compare 5 times
    midResultList = []
    for i in range( len(output_test) ):          #prelearned size, will have total number of output_test
        curResults = []
        for j in range(len(predictingList)):           # prelearned size, the size of elements in curResult
            curPrediction = predictingList[ j ][ i ]
            curResults.append( curPrediction )
        mediumValue = np.median( curResults )        #get the medium value of different predictions in the list
        midResultList.append(mediumValue)
    return midResultList

def ELMprediction(input_train, input_test, output_train,output_test):
    # predicting process, need to get the output_predict 5 times and get the mid of the data for each data
    predictTimeCount = 0
    predictTopLimitCount = 5
    predictingList = []
    while predictTimeCount < predictTopLimitCount:
        output_predict, breakIndicator = process_and_predict(input_train, input_test, output_train, output_test)
        if breakIndicator is False:  # only counting the result if the breakIndicator is false, only accept valid models
            predictTimeCount = predictTimeCount + 1
            predictingList.append(output_predict)
    midResultList = getMidResult(predictingList, output_test)
    getCorrectionRate1(midResultList, output_test)

def gridSearchPrediction(input_train, input_test, output_train,output_test):
    #the gamma value and learing_rate values r important in SVM,it can decide the speed of your processing and the relative accuracy of your program
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] #  #[1e3, 5e3, 1e4, 5e4, 1e5, 5e5 ,1e6] for testing only, can change the learning_rate if wanted
    gamma = [1, 0.1, 0.01, 0.001] #[0.0001, 0.0005, 0.001, 0.005,0.01, 0.1]                    #changing the standards here can speed up a littlt
    myGridSearch = gridSearchSvm(learning_rate, gamma)  #create a gridsearch machine
    gridSearchPredictModel = myGridSearch.trainSVM(input_train,output_train)
    predictingResult = gridSearchPredictModel.predict(input_test)
    getCorrectionRate1(predictingResult, output_test)

def duplicateElementChecker(bestOutputList, checkingElement):   #this function help checking if the element inside of the bestOutputList duplicate with the checkingElement
    duplicateOutputIndicator = False
    for outputInList in bestOutputList:
        if outputInList == checkingElement:  # changing the duplicateIndicator into true, don't append again
            duplicateOutputIndicator = True
            break
    return duplicateOutputIndicator

def checkNAddInputNOutput(previousTempOutput, bestFitTrainingDataOutputs, bestFitTrainingDataInputs , bestOutputList,bestInpuList):
    for i in range(len( previousTempOutput)):  # checking if the preTempOuput has the same output values as the current bestFitTraininDataOutput
        for j in range(len( bestFitTrainingDataOutputs )):
            if previousTempOutput[i] == bestFitTrainingDataOutputs[j]:  # check if the output has already been in bewtOutputList, if not, adding it into the list
                if len( bestOutputList) == 0:  # situation where no output has been add to the bestOutputList yet, add the intput and output in to the bestInput/outputList
                    bestOutputList.append(bestFitTrainingDataOutputs[j])
                    bestInpuList.append(bestFitTrainingDataInputs[j])
                else:  # situation where there are some elements in the bestOutputList, check if duplicate, if not adding the result into the bestInput/outputList
                    duplicateOutputIndicator = duplicateElementChecker(bestOutputList, bestFitTrainingDataOutputs[j])
                    if duplicateOutputIndicator == False:
                        bestOutputList.append(bestFitTrainingDataOutputs[j])
                        bestInpuList.append(bestFitTrainingDataInputs[j])
    return bestOutputList, bestInpuList

def bestInNOutputSingleTimeChecker(myGA_ELM):
    loopCount = 0  #to count how many number of loops the while loop run
    curBestOutList =[]; curBestInpuList=[]                 #to store all the bestOutputs in theory
    previousTempOutput=[]     #store the elements that has duplicate twice each time， also need to check the bestOutputList
    loopingLimit = 20
    while loopCount < loopingLimit:
        originalOutputSubgroups, originalPredictionSubgroups, originalInputSubgroups = myGA_ELM.select()
        winningSubGroupIndexList, tempSubGroupIndexList = myGA_ELM.crossOver(originalOutputSubgroups,originalPredictionSubgroups)
        curBestFitTrainingDataInputs, curBestFitTrainingDataOutputs = myGA_ELM.mutation(winningSubGroupIndexList,tempSubGroupIndexList,originalOutputSubgroups,originalPredictionSubgroups,originalInputSubgroups)
        if loopCount ==0:   #if it is the first time of looping, only appending to the previousTempOutputList
            previousTempOutput= curBestFitTrainingDataOutputs
        else:               #not the first time looping, checking if the current time bestFitTraininigDataOutput duplicates with the prevous time
            curBestOutputList, curBestInpuList = checkNAddInputNOutput(previousTempOutput, curBestFitTrainingDataOutputs, curBestFitTrainingDataInputs, curBestOutList, curBestInpuList)
            previousTempOutput = curBestFitTrainingDataOutputs    #update the previousTempOutput
        loopCount = loopCount + 1
    return curBestInpuList, curBestOutputList, myGA_ELM

def GA_ELM_findweightNThreash():  # this function will checking the surviving generation after GA algorithm for multiple times, and return the best fit input/and output data
    myGA_ELM = GA_ELMPredictor(input_train, output_train, 100)
    LoopCount = 0
    bestInpuList=[]; bestOutputList=[]; prevBestOutputList=[]
    loopingLimit = 4
    while LoopCount < loopingLimit:              #do the single time checker multiple times to prune out more unrelated data Instances, and eventually get the closest bestInput/outputList
        curBestInpuList, curBestOutputList, myGA_ELM = bestInNOutputSingleTimeChecker( myGA_ELM )
        if LoopCount == 0:    #first time of looping
            prevBestOutputList = curBestOutputList
        else:                              #situation to check the preBestOutput duplicate with the curBestOutputSituation
            bestOutputList, bestInpuList = checkNAddInputNOutput(prevBestOutputList, curBestOutputList, curBestInpuList , bestOutputList,bestInpuList)
            prevBestOutputList = bestOutputList  #curBestInpuList   #updating the previousBestOutputList with the besOutputList, keep comparing the existing bestOutput with the new incoming cuBestOutput
        LoopCount = LoopCount + 1            #update the loopCount
    print('bestOutputlist are ', bestOutputList, '\nthe length of the bestOutputList is', len(bestOutputList))
    return  bestInpuList, bestOutputList,myGA_ELM     #bestFitTrainingDataInputs, bestFitTrainingDataOutputs,myGA_ELM

# Press the green button in the gutter to run the script.
if __name__ == '__main__':           #to think of the improvement,could go thinking about normalization, neuronN or activation function
    #getting the example data from  https://blog.csdn.net/rocling/article/details/85239441
    #sample for testing
    x,y = datasets.load_diabetes(return_X_y=True)
    input_train, input_test, output_train, output_test = train_test_split(x, y, test_size=0.2)

    #ELMprediction                 #less stable, but faster
    '''
    #ELMprediction(input_train, input_test, output_train, output_test)
    process_and_predict(input_train, input_test, output_train,output_test)   #only to this step is fine, the other option is the refined version
    '''

    #svmPrediction(gridsearch)      #more stable but slower
    '''
    gridSearchPrediction(input_train, input_test, output_train, output_test)
    '''

    #GA_ELMPrector
    #6'''

    bestFitTrainingDataInputs, bestFitTrainingDataOutputs,myGA_ELM =GA_ELM_findweightNThreash()
    backMappingBestTrainOutputs = myGA_ELM.backMappingFunction(max(output_train), min(output_train), bestFitTrainingDataOutputs)   #note the max output here can also be modified into the global max value of the outputs, might need to change later
    startingWeight, startingThreash = myGA_ELM.findStartWeightNThreash(bestFitTrainingDataInputs, backMappingBestTrainOutputs)
    #after getting the startingWeight and startingThreash, putting it into the ELM machine and rerun

    testingElm = myELM(input_train.shape[1], 1,100)  # reshape the input into a matrix of (500,1), output is 1 and neuronsize is 100
    testingElm.updateStartingWeightNThreash(startingWeight, startingThreash)    #update the ELM machine's weight and threash
    testingElm.train(input_train, output_train.reshape(-1, 1))  # train data
    output_test_predict = testingElm.predictResult(input_test)
    getCorrectionRate1(output_test_predict, output_test)

    #'''

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
