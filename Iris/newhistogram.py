import numpy as np

from importdata import importdata
from MSEtrainer import WFromTraining
from MSEtrainer import sigmoid

samplesAndSolutions = importdata(nsamples = 150, nclasses = 3, nfeatures = 4, filePath = 'Data/iris.data')

samples = samplesAndSolutions[0]
solutions = samplesAndSolutions[1]

trainingClass1 = (samples[0:30,:]).T
trainingClass2 = (samples[50:80,:]).T
trainingClass3 = (samples[100:130,:]).T

testingClass1 = (samples[30:50,:]).T
testingClass2 = (samples[80:100,:]).T
testingClass3 = (samples[130:150,:]).T

print(trainingClass1[0,:])
print('\n')
print(trainingClass2[0,:])
print('\n')
print(trainingClass3[0,:])
print('\n')

plt.hist(testArray, bins)
plt.show()



