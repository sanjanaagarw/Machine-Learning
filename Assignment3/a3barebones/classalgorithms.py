from __future__ import print_function
from __future__ import print_function
from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
import random


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest


class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """

    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(
            np.add(np.dot(Xtrain.T, Xtrain) / numsamples, self.params['regwgt'] * np.identity(Xtrain.shape[1]))),
            Xtrain.T), yt) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest


class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': True}
        if parameters is not None:
            self.getparams()['usecolumnones'] = parameters['usecolumnones']
        self.x_Class0 = None
        self.x_Class1 = None

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        if not self.getparams()['usecolumnones']:
            Xtrain = Xtrain[:, :-1]
        # print("Xtrain shape when useColumns is", self.getparams()['usecolumnones'], Xtrain.shape[1])
        noOfFeatures = Xtrain.shape[1]
        noOfSamples = len(ytrain)

        self.x_Class0 = []
        self.x_Class1 = []
        for i in range(noOfSamples):
            # print(ytrain[i])
            if ytrain[i] == 0:
                # print(i,"y=0")
                self.x_Class0.append(Xtrain[i])
            else:
                # print(i,"y=1")
                self.x_Class1.append(Xtrain[i])

        self.x_Class0 = np.asarray(self.x_Class0).reshape(len(self.x_Class0), Xtrain.shape[1])
        self.x_Class1 = np.asarray(self.x_Class1).reshape(len(self.x_Class1), Xtrain.shape[1])
        # print ("X_Class0.shape",self.x_Class0.shape)
        # print ("X_Class1.shape",self.x_Class1.shape)
        self.mean_Class0 = utils.mean(self.x_Class0)
        self.std_Class0 = utils.stdev(self.x_Class0)

        self.mean_Class1 = utils.mean(self.x_Class1)
        self.std_Class1 = utils.stdev(self.x_Class1)

        # print("mean_Class0.shape", self.mean_Class0.shape)
        # print("std_Class0.shape", self.std_Class0.shape)
        # print("mean_Class1.shape", self.mean_Class1.shape)
        # print("std_Class1.shape", self.std_Class1.shape)
        self.ymean_Class1 = utils.mean(ytrain)
        self.ymean_Class0 = 1 - self.ymean_Class1

        # print("ymean_Class1.shape", type(self.ymean_Class1))
        # print("ymean_Class0.shape", type(self.ymean_Class0))

    def predict(self, Xtest):
        # noOfFeatures = Xtest.shape[1]
        if not self.getparams()['usecolumnones']:
            Xtest = Xtest[:, :-1]
        ytest = np.zeros(Xtest.shape[0])
        for i in range(0, Xtest.shape[0]):
            prob_Class0 = self.ymean_Class0
            prob_Class1 = self.ymean_Class1
            for j in range(0, len(self.mean_Class0)):
                prob_Class0 = prob_Class0 * utils.calculateprob(Xtest[i][j], self.mean_Class0[j], self.std_Class0[j])
                prob_Class1 = prob_Class1 * utils.calculateprob(Xtest[i][j], self.mean_Class1[j], self.std_Class1[j])
            if prob_Class0 < prob_Class1:
                ytest[i] = 1
        return ytest


class LogitReg(Classifier):
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
            self.regularizer_lambda = 0.01
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
            self.regularizer_lambda = 0.01
        elif self.params['regularizer'] is 'elasticNet':
            # TODO: Implement l1-l2
            self.regularizer = (utils.dl1, utils.dl2)
            self.regularizer_lambda1 = 0.01
            self.regularizer_lambda2 = 0.01
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape, ))

    # TODO: implement learn and predict functions

    def learn(self, Xtrain, ytrain, alpha=0.001):
        # self.weights = np.zeros(Xtrain.shape[1])
        noOfFeatures = Xtrain[0].shape[0]
        self.weights = np.random.rand(noOfFeatures)
        noOfSamples = Xtrain.shape[0]
        minCost = utils.computeCost(self.weights, Xtrain, ytrain)
        # incr = 1
        randomIndices = range(noOfSamples)
        count = 1
        epochs = 9
        for incr in range(epochs):
            np.random.shuffle(randomIndices)
            for i in randomIndices:
                oldMinCost = minCost
                x = Xtrain[i]
                y = ytrain[i]
                hypothesis = utils.sigmoid(np.dot(x, self.weights.T))
                loss = hypothesis - y
                # loss = hypothesis - y
                # self.weights =
                # print ("self weights is", self.weights)
                # print ("type", type(self.weights))
                # print ("is self weights vector?", isinstance(self.weights, type(self.weights)))

                if self.params['regularizer'] is 'l1':
                    # print ("For l1:", self.regularizer_lambda)
                    self.weights -= alpha * ((np.dot(loss, x)) - ((self.regularizer_lambda)
                                                                  * utils.dl1(self.weights)))
                    minCost = utils.computeCost(self.weights, x, y)
                    count = oldMinCost - minCost
                elif self.params['regularizer'] is 'l2':
                    # print ("For l2:", self.regularizer_lambda)
                    self.weights -= alpha * ((np.dot(loss, x)) - ((self.regularizer_lambda)
                                                                  * utils.dl2(self.weights)))
                    minCost = utils.computeCost(self.weights, x, y)
                    count = oldMinCost - minCost
                elif self.params['regularizer'] is 'elasticNet':
                    self.weights -= alpha * ((np.dot(loss, x)) - ((self.regularizer_lambda1 * utils.dl1(self.weights)) +
                                                                  (self.regularizer_lambda2 * utils.dl2(self.weights))))
                    minCost = utils.computeCost(self.weights, x, y)
                    count = oldMinCost - minCost
                else:
                    self.weights -= alpha * (np.dot(loss, x))
                    minCost = utils.computeCost(self.weights, x, y)
                    count = oldMinCost - minCost
                    # print ("No of samples", noOfSamples)

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(ytest)
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest


class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = {'nh': 4, 'stepsize': 0.01, 'epochs': 10, 'ni': 9}
        self.nh = self.params['nh']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid
        self.stepsize = self.params['stepsize']
        self.epochs = self.params['epochs']

        self.ni = self.params['ni']
        self.no = 1
        self.input_weights = None
        self.wo = None

    def learn(self, Xtrain, ytrain):

        self.ni = Xtrain.shape[1]
        self.no = 1
        self.nh = self.params['nh']
        self.stepsize = self.params['stepsize']
        self.epochs = self.params['epochs']

        self.wi = np.asarray(self.createMatrix(self.nh, self.ni))
        self.wo = np.asarray(self.createMatrix(self.no, self.nh + 1))

        for incr in range(self.epochs):
            for i in range(Xtrain.shape[0]):
                self.update(Xtrain[i, :], ytrain[i])

    def createMatrix(self, row, col):
        matrix = []
        for i in range(row):
            matrix.append([random.uniform(-1, 1)] * col)
        return matrix

    def update(self, x, y):
        [ah, yHat] = self.evaluate(x)

        x = np.reshape(x.T, (1, self.ni))
        z = np.dot(self.wi, x.T)

        delta = (-y / yHat + (1 - y) / (1 - yHat)) * yHat * (1 - yHat)
        delta1 = delta * np.multiply(self.wo[:, 1:].T, self.dtransfer(z))

        self.wo -= self.stepsize * delta * ah
        self.wi -= self.stepsize * np.dot(delta1, x)

    def evaluate(self, inputs):
        """ Including this function to show how predictions are made """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')

        ah = self.transfer(np.dot(self.wi, inputs))
        ah = np.append([1], ah)

        ao = self.transfer(np.dot(self.wo, ah))

        return ah, ao

    def predict(self, Xtest):
        ah = utils.sigmoid(np.dot(Xtest, self.wi.T))
        ah = np.hstack((np.ones((ah.shape[0], 1)), ah))

        ytest = utils.sigmoid(np.dot(ah, self.wo.T))
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0

        return ytest


class LogitRegAlternative(Classifier):
    def __init__(self, parameters={}):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        noOfSamples = len(Xtrain)
        noOfFeatures = len(Xtrain[0])
        self.weights = np.random.rand(noOfFeatures)
        # print ("Self weights", self.weights)

        epochs = 9
        for incr in range(epochs):
            for i in range(noOfSamples):
                x = Xtrain[i]
                y = ytrain[i]
                hypothesis = self.evaluate(x)
                loss = hypothesis - y
                alpha = 0.01
                self.weights -= alpha * (np.dot(loss, x) * self.cosine(np.dot(self.weights, x)))

    # TODO: implement learn and predict functions
    def evaluate(self, x):
        w = np.dot(x, self.weights)
        return 0.5 * (1 + (w / math.sqrt(1 + w ** 2)))

    def cosine(self, w):
        return 1 / (math.sqrt(1 + (w ** 2)))

    def predict(self, Xtest):
        noOfSamples = len(Xtest)
        ytest = np.ndarray((noOfSamples,))

        for i in range(noOfSamples):
            y = self.evaluate(Xtest[i])
            ytest[i] = int(y > 0.5)
        # print ("Ytest", ytest)
        return ytest
