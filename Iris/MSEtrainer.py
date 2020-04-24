import numpy as np
import math

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def WFromTraining(nclasses, trainingData, solution, maxiter = 100, alpha = 0.1): # training data must be on the format ???
    dataDims = np.shape(trainingData)
    nsamples = dataDims[0]
    nfeatures = dataDims[1]

    paddedData = (np.append((trainingData.T),np.ones(1,nsamples))).T # add ones to the end of the feature samples for the modified form

    W = np.ones((nclasses, nfeatures + 1))# Initialize W (randomly, ones, identity?). Not the added column for the modified form
    g = np.empty(nsamples,nclasses)
    for totalIt in range(maxiter): #perharps also another termination criteria?
        for sampleIt in range(nsamples):
            xCurrent = W*((paddedData[sampleIt,:]).T) # z_i = W * x_i
            for classIt in range(nclasses):
                g[sampleIt, classIt] = sigmoid(xCurrent[classIt]) # g_i = sigmoid(z_i)
    print('heyoo')