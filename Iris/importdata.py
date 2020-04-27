import numpy as np

def importdata(nsamples, nclasses, nfeatures, filePath = 'Data/iris.data'):

    samples = np.empty((nsamples,nfeatures))
    answers = np.zeros((nsamples,nclasses))

    classes = []

    with open(filePath,'r') as file:
        sampleIt = 0
        for line in file:
            lineSplit = line.split(',')
            for fltIt in range(nfeatures):
                samples[sampleIt,fltIt] = float(lineSplit[fltIt])

            classSol = lineSplit[nfeatures]
            if not classSol in classes:
                classes.append(classSol)
            answers[sampleIt,classes.index(classSol)] = 1
            
            sampleIt += 1

    return samples, answers # Returns tuple of two numpy arrays