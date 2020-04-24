import numpy as np
import math

def sigmoid(x):
    return 1/(1 + np.exp(-x))

maxiter = 300

WFromTraining(maxiter = 100, alpha = 0.1, nclass, trainingData, solution): # training data must be on the format ???
    # nsamples = ??
    # nfeatures = ??
    # Initialize W (randomly, ones, identity?)
    for it in range(300): #perharps also another termination criteria?
