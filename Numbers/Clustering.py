import numpy as np
import matplotlib.pyplot as plt

nSamples = 6000
nTests = 10
nClasses = 10

with open('Data/train_images.bin','rb') as binaryFile:
    imgB = binaryFile.read()

with open('Data/train_labels.bin','rb') as binaryFile:
    lbB = binaryFile.read()

with open('Data/test_images.bin','rb') as binaryFile:
    tstimgB = binaryFile.read()

with open('Data/test_labels.bin','rb') as binaryFile:
    tstlbB = binaryFile.read()

img = np.reshape(np.frombuffer(imgB[16:16+784*nSamples], dtype=np.uint8), (nSamples,784))
lb = np.frombuffer(lbB[8:nSamples+8], dtype=np.uint8)

tstimg = np.reshape(np.frombuffer(tstimgB[16:16+784*nSamples], dtype=np.uint8), (nSamples,784))
tstlb = np.frombuffer(tstlbB[8:nSamples+8], dtype=np.uint8)

ref = np.zeros((nClasses,784))

for sampleIt in range(nSamples):
    ref[lb[sampleIt]] += img[sampleIt] / (nSamples/10)

eclDst = np.empty(nClasses)

tstAns = np.empty(nTests)

for sampleIt in range(nTests):
    for classIt in range(nClasses):
        devFromRef = tstimg[sampleIt] - ref[classIt]
        eclDst[classIt] = np.sum(np.multiply(devFromRef,devFromRef))
    tstAns[sampleIt] = np.argmin(eclDst)

def confMatrix(tstAns,tstSol):
    nSamples = (np.shape(tstAns))[0]
    nClasses = tstSol.max()
    confMatrix = np.zeros((nClasses+1,nClasses+1))
    nMisses = 0
    missIndices = np.array([])
    for sampleIt in range(nSamples):
        currentAnswer = int(tstAns[sampleIt])
        currentSolution = int(tstSol[sampleIt])
        confMatrix[currentSolution,currentAnswer] += 1
        if currentAnswer != currentSolution:
            nMisses += 1
            missIndices = np.append(missIndices, sampleIt)

    return confMatrix, nMisses/nSamples, missIndices

def accumDist(samples,clusters,sampleOwnership):    # For one class
    dist = 0
    dataDims = np.shape(samples)
    nSamples = dataDims[0]
    for sampleIt in range(nSamples):
        currentOwner = sampleOwnership[sampleIt]
        devFromRef = samples[sampleIt] - clusters[currentOwner]
        dist += np.sum(np.multiply(devFromRef,devFromRef))

    return dist


def clusters(img, lb, maxClusters = 64):
    dataDims = np.shape(img)
    nSamples = dataDims[0]

    nClasses = int(lb.max()+1)

    # solDims = np.shape(lb)
    # nClasses = solDims[0]

    clusters = 500 * np.ones((maxClusters*nClasses,784)) # To avoid classifying a sample to a 'uninitialized' cluster, ensure large distance
    clusterSol = np.empty(maxClusters*nClasses)
    for classIt in range(nClasses):
        for clusterIt in range(maxClusters):
            clusterSol[maxClusters*classIt + clusterIt] = classIt   # The label corresponding to the clusters we will end up with

    sortedImg = sortedImage(img,lb)
    _, classCount = np.unique(lb, return_counts = True)

    startIndex = 0

    for classIt in range(nClasses): # Cluster creation for each class individually
        classLength = classCount[classIt]   # Amount of samples for current class
        print('Current class is : ', classIt, ' With ', classLength, ' samples of the current class')
        currentSamples = sortedImg[startIndex:startIndex+classLength]  # Samples for current class
        currentClusters = clusters[maxClusters*classIt:maxClusters*classIt+maxClusters] # Clusters for current class
        
        # Alg from compendium start

        for sampleIt in range(classLength):
            currentClusters[0] += currentSamples[sampleIt]
        currentClusters[0] = currentClusters[0] / classLength   # Creating first cluster/reference

        for amountOfClustersIt in range(1,maxClusters+1):   # We start out with one cluster and end with maxClusters
            prevAccDist = 123456789 # Arbitrary, just needs to be big
            mostPopularCluster = 0  # The cluster with the most associated samples
            while True: # Contains the steps 3,4,5
                sampleOwnership = np.empty(classLength,dtype=np.int) # List contains the cluster that the index belongs to : array[sample] = cluster
                eclDst = np.empty(maxClusters) # A samples distances to the clusters
                for sampleIt in range(classLength): # Iterate over current class' samples
                    for clusterIt in range(maxClusters): # Iterate over the clusters
                        devFromRef = [sampleIt] - ref[classIt]
                        eclDst[clusterIt] = np.sum(np.multiply(devFromRef,devFromRef))
                    currentSampleCluster = np.argmin(eclDst) # What cluster the current sample belongs to
                    sampleOwnership[sampleIt] = currentSampleCluster
                currAccDist = accumDist(currentSamples,currentClusters,sampleOwnership)

                if (currAccDist/prevAccDist) > 0.98:
                    break

                # clusterFq = np.bincount(sampleOwnership)    # The amount of samples classified with each cluster : array[cluster] = nsamples

                maxHits = 0
                
                for clusterIt in range(amountOfClustersIt):
                    # currentClusterFq = clusterFq[clusterIt] # Frequency of matching with current cluster
                    currentClusters[clusterIt] = np.zeros(784) # Reset cluster mean values so we can recalculate it
                    hits = 0    # Amount of samples classified with current cluster
                    
                    for sampleIt in range(classLength):
                        if sampleOwnership[sampleIt] == clusterIt:
                            currentClusters[clusterIt] += currentSamples[sampleIt] # Add to mean
                            hits += 1   # Found a sample that matched with our cluster
                            if hits > maxHits:
                                maxHits = hits
                                mostPopularCluster = clusterIt
                    currentClusters[clusterIt] = currentClusters[clusterIt] / hits  # Finally calculate mean by dividing by amount of samples

            if amountOfClustersIt < maxClusters:
                currentClusters[amountOfClustersIt] = currentClusters[mostPopularCluster] + (0.2*np.random.rand(784)-0.1) # We 'split' : we get a new cluster

        for clusterIt in range(maxClusters):    # Stitch back together the clusters one class at a time
            clusters[maxClusters*classIt+clusterIt] = currentClusters[clusterIt]

        # Alg from compendium end

        startIndex += classLength

    return clusters, clusterSol

def sortedImage(img,lb):
    dataDims = np.shape(img)
    nSamples = dataDims[0]

    solDims = np.shape(lb)
    nClasses = solDims[0]

    sortedImg = np.empty((nSamples,784))    # Same length as sample array as it has the same values in a different order
    classIndices = np.empty(0)
    for classIt in range(nClasses):
        classIndices = np.append(classIndices,np.argwhere(lb == classIt))

    for sampleIt in range(nSamples):
        sortedImg[sampleIt] = img[int(classIndices[sampleIt])]

    return sortedImg



clusters, clusterSol = clusters(img,lb,maxClusters=4)

nClusters = (np.shape(clusters))[0]

eclDst = np.empty(nClusters)

match = np.empty(nTests)

tstAns = np.empty(nTests)

for testIt in range(nTests):
    for sampleIt in range(nClusters):
        devFromSample = tstimg[testIt] - clusters[sampleIt]
        eclDst[sampleIt] = np.sum(np.multiply(devFromSample,devFromSample))
    closestMatch = np.argmin(eclDst)
    match[testIt] = closestMatch
    tstAns[testIt] = lb[closestMatch]

answerPlt = np.reshape(tstimg[7],(28,28))
solutionPlt = np.reshape(clusters[int(match[7])],(28,28))

plt.suptitle('??? classified number')

plt.subplot(1,2,1)
plt.imshow(answerPlt,cmap='gray',vmin=0,vmax=255)
plt.title('This test image matched')

plt.subplot(1,2,2)
plt.imshow(solutionPlt,cmap='gray',vmin=0,vmax=255)
plt.title('With this reference image')

plt.show()

# print('Our answers are : \n', tstAns[0:24])
# print('The real answers are : \n', tstlb[0:24])

# plt.imshow(np.reshape(tstimg[11],(28,28)),cmap='gray',vmin=0,vmax=255)

# plt.show()

# ref0 = np.reshape(ref[0],(28,28))
# ref1 = np.reshape(ref[1],(28,28))
# ref2 = np.reshape(ref[2],(28,28))
# ref3 = np.reshape(ref[3],(28,28))
# ref4 = np.reshape(ref[4],(28,28))
# ref5 = np.reshape(ref[5],(28,28))
# ref6 = np.reshape(ref[6],(28,28))
# ref7 = np.reshape(ref[7],(28,28))
# ref8 = np.reshape(ref[8],(28,28))
# ref9 = np.reshape(ref[9],(28,28))

# plt.subplot(5,2,1)
# plt.imshow(ref0,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,2)
# plt.imshow(ref1,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,3)
# plt.imshow(ref2,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,4)
# plt.imshow(ref3,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,5)
# plt.imshow(ref4,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,6)
# plt.imshow(ref5,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,7)
# plt.imshow(ref6,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,8)
# plt.imshow(ref7,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,9)
# plt.imshow(ref8,cmap='gray',vmin=0,vmax=255)

# plt.subplot(5,2,10)
# plt.imshow(ref9,cmap='gray',vmin=0,vmax=255)

# plt.show()