from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import division  # floating point division
import math
import numpy as np

import classalgorithms as algs
from sklearn.metrics import classification_report

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct / float(len(ytest))) * 100.0


def geterror(ytest, predictions):
    return 100.0 - getaccuracy(ytest, predictions)

'''Source for k_cross_validation: http://code.activestate.com/recipes/521906-k-fold-cross-validation-partition/'''


def k_cross_validation(X, K, randomise=False):
    if randomise:
        from random import shuffle
        X = list(X)
        shuffle(X)
    for k in xrange(K):
        training = [x for j, x in enumerate(X) if j % K != k]
        validation = [x for j, x in enumerate(X) if j % K == k]
        yield training, validation


if __name__ == '__main__':
    dataset = np.genfromtxt('datasets/numericsequence.csv', delimiter=',')
    # dataset = dtl.load_occupancy_dataset()
        # dtl.load_occupancy_dataset(trainsize, testsize)
    np.random.shuffle(dataset)
    errors = {}
    # accuracies = []
    classalgs = {}
    numparams = 0
    parameters = {}
    for i in range(5):
        smallDataSet = dataset[i*1300:(i+1)*1300]
        classalgs = {
            'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
            'Logistic Regression': algs.LogitReg(),
            'Neural Network': algs.NeuralNet({'epochs': 100, 'stepsize': 0.01, 'nh': 8, 'ni': 19})
            # 'L1 Logistic Regression': algs.LogitReg({'regularizer': 'l1'}),
            # 'L2 Logistic Regression': algs.LogitReg({'regularizer': 'l2'}),
        }
        numalgs = len(classalgs)
        parameters = (
            # {'regwgt': 0.0, 'nh': 4},
            {'regwgt': 0.01, 'nh': 8},
            # {'regwgt': 0.05, 'nh': 16},
            # {'regwgt': 0.1, 'nh': 32},
        )
        numparams = len(parameters)


        for learnername in classalgs:
            errors[learnername] = np.zeros((numparams, 5))
            # print ("For learner name" + learnername)
            # + " the errors[] is \n" + str(errors[learnername]))
        for learnername, learner in classalgs.iteritems():

            for trainset, testset in k_cross_validation(smallDataSet, K=10):
                trainset = np.array(trainset)
                testset = np.array(testset)
                numberOfInputs = -1
                Xtrain = trainset[:, 0:numberOfInputs]
                ytrain = trainset[:, numberOfInputs]
                Xtest = testset[:, 0:numberOfInputs]
                ytest = testset[:, numberOfInputs]

                Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0], 1))))
                Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0], 1))))

                trainset = (Xtrain, ytrain)
                testset = (Xtest, ytest)

                for p in range(numparams):
                    params = parameters[p]
                    # Reset learner for new parameters
                    learner.reset(params)
                    print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                    # Train model
                    # print("Start learning")
                    learner.learn(trainset[0], trainset[1])
                    # print ("Done learning")
                    # Test model
                    # print("Start predicting")
                    predictions = learner.predict(testset[0])
                    # print("Predictions\n", predictions)
                    error = geterror(testset[1], predictions)
                    accuracy = getaccuracy(testset[1], predictions)
                    # accuracies[count] += (accuracy)

                    print(classification_report(testset[1], predictions))
                    # Accuracy Without using scikit.
                    print("The accuracy score for " + learnername + " is " + str(accuracy))
                    # Accuracy Using scikit learn.
                    # print("The accuracy score is {:.2%}".format(accuracy_score(testset[1], predictions)))
                    print('Error for ' + learnername + ': ' + str(error))
                    # print('Accuracy for ' + learnername + ': ' + str(100 - error))
                    errors[learnername][p, i] = error
                    # accuracies[learnername][p, i] = accuracy

    for learnername, learner in classalgs.iteritems():
        besterror = np.mean(errors[learnername][0, :])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p, :])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(
            1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(5)))
