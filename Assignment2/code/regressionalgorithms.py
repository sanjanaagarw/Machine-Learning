from __future__ import division  # floating point division
import numpy as np
import time

import utilities as utils


def l2err(prediction, ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))


def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / ytest.shape[0]


class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    
    def __init__(self, params={}):
        """ Params can contain any useful parameters for the algorithm """
        self.weights = None
        self.params = {}
        
    def reset(self, params):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,params)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
        # Could also add re-initialization of weights, so that does not use previously learned weights
        # However, current learn always initializes the weights, so we will not worry about that
        
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """        
        ytest = np.dot(Xtest, self.weights)
        return ytest


class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    
    def __init__(self, params={}):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        self.params = {}
                
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest


class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, params={}):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean
        

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5]}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        print (numsamples)
        Xless = Xtrain[:, self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T, Xless)), Xless.T), ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)       
        return ytest


class StochasticRegression(Regressor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'epoch': 1, 'incr': 0.01}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        self.weights = np.random.rand(Xless.shape[1])
        incr = self.getparams()['incr']
        epoch = self.getparams()['epoch']
        for item in range(epoch):
            for i in range(0, numsamples):
                x = Xless[i]
                y = ytrain[i]
                gt = np.dot(((np.dot(x.T, self.weights)) - y), x)
                self.weights -= (incr * gt)

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest


class BatchRegression(Regressor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'B': 0.5, 'tolerance': 10e-4, 'initStep': 1}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        start = time.time()
        """ Learns using the traindata """
        numsamples = Xtrain.shape[0]
        print ("Number of samples", numsamples)
        Xless = Xtrain[:, self.params['features']]

        self.weights = np.zeros(Xless.shape[1])
        print ("xless shape 1", Xless.shape[1])
        print ("self weights", self.weights)
        B = self.params['B']
        tolerance = self.params['tolerance']

        xtranspose_x = np.dot(Xless.T, Xless)
        xtranspose_x /= numsamples

        Xy = np.dot(Xless.T, ytrain)
        Xy /= numsamples

        incr = self.params['initStep']

        predictions = self.predict(Xless)
        dataError = geterror(predictions, ytrain)
        count = 0
        while True:
            count += 1
            error = dataError
            incr = self.getIncrementValue(incr, Xless, ytrain, Xy, B)
            self.weights = self.weights-incr * (np.dot(xtranspose_x, self.weights)-Xy)
            predictions = self.predict(Xless)
            dataError = geterror(predictions, ytrain)
            if np.abs(dataError - error) < tolerance:
                break
        print ("Epoch number is:" + count)
        print ("Time:" + (time.time() - start))

    def getIncrementValue(self, oldIncr, Xless, ytrain, Xy, B):
        w1 = self.weights - oldIncr * (np.dot(oldIncr, self.weights) - Xy)
        dataErrorNew = geterror(np.dot(Xless, w1), ytrain)

        predictions = self.predict(Xless)
        dataErrorOld = geterror(predictions, ytrain)

        if dataErrorNew > dataErrorOld:
            return self.getIncrementValue(oldIncr * B, Xless, ytrain, Xy, B)
        else:
            return oldIncr

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class LassoRegression(Regressor):

    def __init__( self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'threshold': 1, 'tolerance': 10e-3}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        numsamples = Xtrain.shape[0]

        Xless = Xtrain[:, self.params['features']]
        threshold = self.getparams()['threshold']

        self.weights = np.zeros(Xless.shape[1])
        tolerance = self.getparams()['tolerance']

        xtranspose_x = np.dot(Xless.T,Xless)
        xtranspose_x /= numsamples

        Xy = np.dot(Xless.T, ytrain)
        Xy /= numsamples
        incr = utils.l2(xtranspose_x)
        incr = 1 / (incr * 2)

        predictions = self.predict(Xless)
        dataError = geterror(predictions, ytrain)
        count = 0
        while True:
            count += 1
            error = dataError
            delta = self.weights - incr * (np.dot(xtranspose_x, self.weights) - Xy)
            self.weights = self.shrinkageOperator(delta, threshold)

            predictions = self.predict(Xless)
            dataError = geterror(predictions, ytrain)
            if np.abs(dataError - error) < tolerance:
                break

    def shrinkageOperator(self, delta, threshold):
        for i in range(delta.shape[0]):
            delta[i] = np.sign(delta[i]) * max(np.abs(delta[i]) - threshold, 0)
        return delta

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest


class RidgeRegression(Regressor):

    def __init__( self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'lambda': 0.01}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        Xless = Xtrain[:, self.params['features']]
        xtranspose_x = np.dot(Xless.T, Xless)
        size = xtranspose_x.shape[0]

        self.weights = np.dot(np.dot(np.linalg.inv(xtranspose_x + self.getparams()
        ['lambda'] * np.identity(size)), Xless.T), ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest


class MatchingPursuitLinearRegression(Regressor):
    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5], 'epsilon': 0.0000001}
        self.reset(params)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        initX = Xless[:, 0:2]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(initX.T, initX)), initX.T), ytrain)
        residual = np.dot(initX, self.weights) - ytrain

        stoppingCondition = 10

        featureList = range(2, numfeatures)
        selectedFeatures = [0, 1]

        while stoppingCondition > ['epsilon']:
            corelationMatrix = []
            for feature in featureList:
                corelationMatrix.append(abs(np.dot(Xless[:, feature].T, residual)))

            index = corelationMatrix.index(max(corelationMatrix))
            stoppingCondition = corelationMatrix[index]

            index = featureList[index]
            selectedFeatures.append(index)
            featureList.remove(index)

            if len(featureList) == 0:
                break
            maxX = Xless[:, index].reshape(1000, 1)
            initX = np.append(initX, maxX, axis=1)

            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(initX.T, initX)), initX.T), ytrain)
            residual = np.dot(initX, self.weights) - ytrain

        params = {'features': selectedFeatures}
        self.reset(params)
        Xless = Xtrain[:, self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T, Xless)), Xless.T), ytrain)

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest