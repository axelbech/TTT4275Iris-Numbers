import numpy as np

from importdata import importdata
from MSEtrainer import WFromTraining
from MSEtrainer import sigmoid

samplesAndSolutions = importdata(nsamples = 150, nclasses = 3, nfeatures = 4, filePath = 'Data/iris.data')

samples = samplesAndSolutions[0]
solutions = samplesAndSolutions[1]

trainingSamples = np.concatenate((samples[0:30,:],np.concatenate((samples[50:80,:],samples[100:130,:]),axis=0)),axis=0)
testingSamples = np.concatenate((samples[30:50,:],np.concatenate((samples[80:100,:],samples[130:150,:]),axis=0)),axis=0)
# First 30 for training, last 20 for testing
trainingSolutions = np.concatenate((solutions[0:30,:],np.concatenate((solutions[50:80,:],solutions[100:130,:]),axis=0)),axis=0)
testingSolutions = np.concatenate((solutions[30:50,:],np.concatenate((solutions[80:100,:],solutions[130:150,:]),axis=0)),axis=0)

W = WFromTraining(trainingSamples, trainingSolutions, alpha=0.01, maxiter=1000)


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

        MSE += 0.5 * (np.dot(((g_k-t_k).T),g_k-t_k))

    return MSE

def confMatrix(testSamples, testSolutions, W):
    dataDims = np.shape(testSamples)
    nsamples = dataDims[0]

    solDims = np.shape(testSolutions)
    nclasses = solDims[1]

    paddedData = (np.concatenate(((testSamples.T),np.ones((1,nsamples))),axis=0)).T
    g = np.zeros((nsamples,nclasses))

    nmisses = 0
    confMatrix = np.zeros((nclasses,nclasses))

    for sampleIt in range(nsamples):
        xCurrent = (np.array(paddedData[sampleIt,:],ndmin=2)).T
        zCurrent = np.dot(W,xCurrent) # z_i = W * x_i
        for classIt in range(nclasses):
            g[sampleIt, classIt] = sigmoid(zCurrent[classIt,:]) # g_i = sigmoid(z_i)

    answers = np.argmax(g,axis=1) # Collection of g's 'answers' at each column (answers is a 1D array)

    for sampleIt in range(nsamples):
        currentAnswer = answers[sampleIt]
        currentSolution = np.argmax(testSolutions[sampleIt,:])
        confMatrix[currentSolution,currentAnswer] += 1
        if currentAnswer != currentSolution:
            nmisses += 1

    return confMatrix, nmisses/nsamples


ourMSE = MSE(testingSamples,testingSolutions,W)

print('MSE from our W = ', ourMSE)

confMerrR = confMatrix(testingSamples,testingSolutions,W)

print('30/20 confusion matrix : \n',confMerrR[0])
print('30/20 error rate : ',confMerrR[1])