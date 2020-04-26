from importdata import importdata
from MSEtrainer import WFromTraining

samplesAndSolutions = importdata(nsamples = 150, nclasses = 3, nfeatures = 4, filePath = 'Data/iris.data')

samples = samplesAndSolutions[0]
solutions = samplesAndSolutions[1]

W = WFromTraining(samples, solutions)