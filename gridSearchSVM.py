import numpy as np
#from sklearn.model_selection import StratifiedKFold #交叉验证 cross verification between testing and training classes
from sklearn.model_selection import GridSearchCV #网格搜索  the classifier we used in the class
#from xgboost import XGBClassifier                     #need to download xgboost, one type of boosting classifier, can help make classifications
from sklearn.svm import SVC
"""
this file will create an svm(supporter machine) class which enalbe users to input some data and some output data and then create a module to predict later outputs
it will project data into higher dimensions, and then make classifications of them. after that, it will give a module which enable to input data and make new predictions of them
please not this class was inspired from 
https://blog.csdn.net/weixin_37450657/article/details/78840831
with some minor changes
"""
class gridSearchSvm(object):
    def __init__(self,C, gamma):
        '''
        this constructor will input the C and gamma value, to create a paramGrid. and then, store the paramGrid for later use of making predictions.
        this constructor will also set up the crossverification set here, later used to divide the trainingset and the testingset
        :param C:
              penalty function, a type of fault tolerant variable,to decide how much faults we can have during the process of mapping
        :param gamma:
              the element to decide what the distribution of data looks like after mapping
        '''
        self.C = C      #fixed way to calling, have to write C,gamma....
        self.gamma = gamma

        # setting up the dictionary for later use of gridsearch
        paramGrid = dict(C = C, gamma =gamma)
        self.paramGrid = paramGrid

        #dividing the training folds into 10(test) and n-10(train), n is the number of total training data, for training purpose, tratifiedKFolder will ensure
        #the sample ratio is the same as the original data ratio

        #gridModule, later to store the module after the training process, will be used for testing
        self.gridModule = 0

    def trainSVM(self,input_train, output_train):
        '''
        this function will allow users to input the training terms and the training results to set up a model for later prediction purpose
        :param input_train:
                    the training terms,normally as matrices
        :param output_train:
                    the training results, normally as matrices
        :return:
                    grid_module, the module after training
        '''
        grid_search = GridSearchCV(SVC(kernel = 'rbf', class_weight = 'balanced'),self.paramGrid)

        grid_module = grid_search.fit(input_train, output_train)  # 运行网格搜索
        self.gridModule = grid_module

        return self.gridModule     #return the module after training

    def predictBySVM(self, input_test):
        '''
        this function will take the tesing input, use the testing input and the module to generate a test output
        :param input_test:
                    a matrix contianing the testing input information
        :return:
                    testing output(output_test)
        '''
        output_test = self.gridModule.predict(input_test)    #might have some issues here, need to check later
        return output_test