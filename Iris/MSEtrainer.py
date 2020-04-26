import numpy as np
import math

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def WFromTraining(trainingData, solution, maxiter = 100, alpha = 0.1): # training data must be on the format ???
    dataDims = np.shape(trainingData)
    nsamples = dataDims[0]
    nfeatures = dataDims[1]

    solDims = np.shape(solution)
    nclasses = solDims[1]

    paddedData = (np.concatenate(((trainingData.T),np.ones((1,nsamples))),axis=0)).T # add ones to the end of the feature samples for the modified form
    print(paddedData)

    W = np.ones((nclasses, nfeatures + 1)) # Initialize W (randomly, ones, identity?). Not the added column for the modified form
    g = np.empty((nsamples,nclasses))
    for totalIt in range(maxiter): # perharps also another termination criteria?
        nabW = np.zeros((nclasses, nfeatures + 1))

        for sampleIt in range(nsamples):
            xCurrent = np.dot(W,(paddedData[sampleIt,:]).T) # z_i = W * x_i
            for classIt in range(nclasses):
                g[sampleIt, classIt] = sigmoid(xCurrent[classIt]) # g_i = sigmoid(z_i)

            g_k = (g[sampleIt,:]).T
            t_k = (solution[sampleIt,:]).T
            x_k = (paddedData[sampleIt,:]).T

            # Sum element [(g_k-t_k)◦g_k◦(1-g_k)]x_k.T
            nabW += np.dot(np.multiply(np.multiply(g_k-t_k,g_k),np.ones((nclasses,1))-g_k),x_k.T)
            # Summed up over nsamples to obtain ∇_W MSE

        W = W - alpha * nabW # W(m) = W(m-1) - α ∇_W MSE
    
    return W # Returns W after terminating