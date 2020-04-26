import numpy as np

from importdata import importdata
from MSEtrainer import WFromTraining
from MSEtrainer import sigmoid

samplesAndSolutions = importdata(nsamples = 150, nclasses = 3, nfeatures = 4, filePath = 'Data/iris.data')

samples = samplesAndSolutions[0]
solutions = samplesAndSolutions[1]

W = WFromTraining(samples, solutions, alpha=0.01, maxiter=1000)


def MSE(testSamples, testSolutions, W):
    dataDims = np.shape(testSamples)
    nsamples = dataDims[0]

    solDims = np.shape(testSolutions)
    nclasses = solDims[1]

    paddedData = (np.concatenate(((testSamples.T),np.ones((1,nsamples))),axis=0)).T
    g = np.zeros((nsamples,nclasses))

    MSE = 0

    for sampleIt in range(nsamples):
        xCurrent = (np.array(paddedData[sampleIt,:],ndmin=2)).T
        zCurrent = np.dot(W,xCurrent) # z_i = W * x_i
        for classIt in range(nclasses):
            g[sampleIt, classIt] = sigmoid(zCurrent[classIt,:]) # g_i = sigmoid(z_i)

        g_k = (np.array((g[sampleIt,:]), ndmin=2)).T
        t_k = (np.array((testSolutions[sampleIt,:]), ndmin=2)).T
        x_k = (np.array((paddedData[sampleIt,:]), ndmin=2)).T

        MSE += 0.5 * (np.dot(((g_k-t_k).T),g_k-t_k))

    return MSE

ourMSE = MSE(samples,solutions,W)

randomMSE = MSE(samples,solutions, 1 * np.random.rand(3, 5) - 0.5)

print('MSE from our W = ', ourMSE)
print('MSE from random W = ', randomMSE)