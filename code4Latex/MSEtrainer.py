import numpy as np
import math

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def WFromTraining(trainingData, solution, maxiter, alpha = 0.1):
    dataDims = np.shape(trainingData)
    nsamples = dataDims[0]
    nfeatures = dataDims[1]

    solDims = np.shape(solution)
    nclasses = solDims[1]

    # add ones to the end of the feature samples for the modified form
    paddedData = (np.concatenate(((trainingData.T),np.ones((1,nsamples))),axis=0)).T 

    # Initialize W. Choice of start W is be crucial
    W = np.array([[0.4, 1.3, -2.0, -1.0, 0.3], [1.0, -2.0, 0.3, -1.4, 0.6], [-1.8, -1.7, 2.8, 2.4, -1.0]])
    g = np.zeros((nsamples,nclasses))
    for totalIt in range(maxiter):
        nabW = np.zeros((nclasses, nfeatures + 1))
        for sampleIt in range(nsamples):
            xCurrent = (np.array(paddedData[sampleIt,:],ndmin=2)).T
            zCurrent = np.dot(W,xCurrent) # z_i = W * x_i
            for classIt in range(nclasses):
                g[sampleIt, classIt] = sigmoid(zCurrent[classIt,:]) # g_i = sigmoid(z_i)

            g_k = (np.array((g[sampleIt,:]), ndmin=2)).T
            t_k = (np.array((solution[sampleIt,:]), ndmin=2)).T
            x_k = xCurrent

            # Sum element [(g_k-t_k)◦g_k◦(1-g_k)]x_k.T
            nabW += np.dot(np.multiply(np.multiply(g_k-t_k,g_k),np.ones((nclasses,1))-g_k),x_k.T)
            # Summed up over nsamples to obtain ∇_W MSE

        W = W - alpha * nabW # W(m) = W(m-1) - α ∇_W MSE
    return W # Returns W after terminating