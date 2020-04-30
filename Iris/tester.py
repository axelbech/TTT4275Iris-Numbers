import numpy as np

from importdata import importdata
from MSEtrainer import WFromTraining
from MSEtrainer import sigmoid

samplesAndSolutions = importdata(nsamples = 150, nclasses = 3, nfeatures = 4, filePath = 'Data/iris.data')

samples = samplesAndSolutions[0]
solutions = samplesAndSolutions[1]

trainingSamples1 = np.concatenate((samples[0:30,:],np.concatenate((samples[50:80,:],samples[100:130,:]),axis=0)),axis=0)
testingSamples1 = np.concatenate((samples[30:50,:],np.concatenate((samples[80:100,:],samples[130:150,:]),axis=0)),axis=0)
# First 30 for training, last 20 for testing
trainingSolutions1 = np.concatenate((solutions[0:30,:],np.concatenate((solutions[50:80,:],solutions[100:130,:]),axis=0)),axis=0)
testingSolutions1 = np.concatenate((solutions[30:50,:],np.concatenate((solutions[80:100,:],solutions[130:150,:]),axis=0)),axis=0)

trainingSamples2 = np.concatenate((samples[20:50,:],np.concatenate((samples[70:100,:],samples[120:150,:]),axis=0)),axis=0)
testingSamples2 = np.concatenate((samples[0:20,:],np.concatenate((samples[50:70,:],samples[100:120,:]),axis=0)),axis=0)
# Last 30 for training, first 20 for testing
trainingSolutions2 = np.concatenate((solutions[20:50,:],np.concatenate((solutions[70:100,:],solutions[120:150,:]),axis=0)),axis=0)
testingSolutions2 = np.concatenate((solutions[0:20,:],np.concatenate((solutions[50:70,:],solutions[100:120,:]),axis=0)),axis=0)



W1 = WFromTraining(trainingSamples1, trainingSolutions1, alpha=0.01, maxiter=2000)
W2 = WFromTraining(trainingSamples2, trainingSolutions2, alpha=0.01, maxiter=2000)


# Histograms

print(trainingSamples1)


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

# First 30 for training, last 20 for testing
#print('First 30 for training, last 20 for testing\nW:\n', W1, '\n')
confMerrR1Training = confMatrix(trainingSamples1,trainingSolutions1,W1)
confMerrR1Test = confMatrix(testingSamples1,testingSolutions1,W1)

#print('First 30/20 confusion matrix : \n', confMerrR1Test[0])
#print('First 30/20 error rate : ', confMerrR1Test[1])

#print('Test\n\nFirst 30/20 confusion matrix : \n', confMerrR1Training[0])
#print('First 30/20 error rate : ', confMerrR1Training[1])


# Last 30 for training, first 20 for testing
#print('\n\nLast 30 for training, first 20 for testing:\nW:\n', W2, '\n')
confMerrR2Training = confMatrix(trainingSamples2,trainingSolutions2,W2)
confMerrR2Test = confMatrix(testingSamples1,testingSolutions1,W2)

#print('Last 30/20 confusion matrix : \n', confMerrR2Test[0])
#print('Last 30/20 error rate : ', confMerrR2Test[1])

#print('Test\n\nFirst 30/20 confusion matrix : \n', confMerrR2Training[0])
#print('First 30/20 error rate : ', confMerrR2Training[1])