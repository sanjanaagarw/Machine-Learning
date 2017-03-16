from __future__ import division  # floating point division
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs


def l2err(prediction, ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))


def l1err(prediction, ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction, ytest), ord=1)


def l2err_squared(prediction, ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))


def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / ytest.shape[0]


if __name__ == '__main__':
    trainsize = 2000
    testsize = 8000
    numruns = 1

    regressionalgs = {
        # 'Random': algs.Regressor(),
        # 'Mean': algs.MeanPredictor(),
        # 'FSLinearRegression': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
        # 'FSLinearRegression': algs.FSLinearRegression({'features': range(69)}),
        # 'StochasticRegression': algs.StochasticRegression({'features': range(69)}),
        # 'RidgeRegression': algs.RidgeRegression({'features': range(69)}),
        # 'LassoRegression': algs.LassoRegression({'features': range(69)}),
        # 'MatchingPursuitLinearRegression': algs.MatchingPursuitLinearRegression({'features': range(1, 68)}),
        'BatchRegression': algs.BatchRegression({'features': range(69)})
    }

    emptyDict = {}
    params = {'Random': [emptyDict],
              'Mean': [emptyDict],
              'FSLinearRegression': [emptyDict],
              'RidgeRegression': [{'lambda': 0.01}, {'lambda': 0.1}, {'lambda': 1}],
              'LassoRegression': [{'threshold': 1, 'tolerance': 10e-3},
                                  {'threshold': 10e-3, 'tolerance': 10e-3},
                                  {'threshold': 10e-4, 'tolerance': 10e-4}],
              'StochasticRegression': [{'epoch': 1, 'incr': 0.01}, {'epoch': 2, 'incr': 0.01},
                                       {'epoch': 3, 'incr': 0.01}, {'epoch': 4, 'incr': 0.01},
                                       {'epoch': 10, 'incr': 0.01}],
              'BatchRegression': [{'B': 0.5, 'tolerance': 10e-6, 'initStep': 0.1},
                                  {'B': 0.5, 'tolerance': 10e-5, 'initStep': 1},
                                  {'B': 0.5, 'tolerance': 10e-4, 'initStep': 1}],
              'MatchingPursuitLinearRegression': [{'epsilon': 0.0000001}, {'epsilon': 0.000001},
                                                  {'epsilon': 0.00001}]
              }
    errors = {}
    meanError = {}
    standardError = {}
    numalgs = len(regressionalgs)

    regressionAlgo = ""
    for learnerName, learner in regressionalgs.iteritems():
        regressionAlgo = learnerName
        errors[learnerName] = np.zeros((len(params[learnerName]), numruns))
        meanError[learnerName] = np.zeros((len(params[learnerName])))
        standardError[learnerName] = np.zeros((len(params[learnerName])))

    print("Regression algorithm we run is: " + regressionAlgo.upper())
    trainingSet = []
    testSet = []
    for i in range(0, numruns):
        trainset, testset = dtl.load_ctscan(trainsize, testsize)
        trainingSet.append(trainset)
        testSet.append(testset)
    print ('Running on train={0} and test={1}'.format(trainset[0].shape[0], testset[0].shape[0]))

    for learnerName, learner in regressionalgs.iteritems():
        for param in range(0, len(params[learnerName])):
            for run in range(0, numruns):
                trainset = trainingSet[run]
                testset = testSet[run]
                # Reset learner, and give new parameters; currently no parameters to specify
                learner.reset(params[learnerName][param])
                # print ('Running ' + learnerName + ' learner on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                # print ('Error on ' + learnerName + ' learner is: ' + str(error))
                errors[learnerName][param, run] = error
                meanError[learnerName][param] += error
            meanError[learnerName][param] /= numruns
            # print ('Mean error on ' + learnerName + ' learner on parameters: ' + str(learner.getparams()))
            # print (meanError[learnerName][param])
            dev = 0
            for run in range(0, numruns):
                dev += np.power((errors[learnerName][param, run] - meanError[learnerName][param]), 2)
            dev = np.sqrt(dev)
            standardError[learnerName][param] = dev / numruns
            # print ('Standard error on ' + learnerName + ' learner on parameters: ' + str(learner.getparams()))
            # print (standardError[learnerName][param])

    print ("Collection of all the errors are:", errors)
    print ("Collection of all the mean errors are:", meanError)
    print ("Collection of all the standard errors are:", standardError)
