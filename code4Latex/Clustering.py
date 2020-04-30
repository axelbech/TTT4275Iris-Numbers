import numpy as np
import matplotlib.pyplot as plt

nSamples = 6000
nTests = 500
nClasses = 10
nClusters = 64

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
tstimg = np.reshape(np.frombuffer(tstimgB[16:16+784*nTests], dtype=np.uint8), (nTests,784))
tstlb = np.frombuffer(tstlbB[8:nTests+8], dtype=np.uint8)

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
        dist += np.linalg.norm(samples[sampleIt] - clusters[currentOwner])
    return dist

def clusters(img, lb, maxClusters = 64):
    dataDims = np.shape(img)
    nSamples = dataDims[0]
    nClasses = int(lb.max()+1)

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
                        # devFromRef = currentSamples[sampleIt] - ref[classIt]
                        # eclDst[clusterIt] = np.sum(np.multiply(devFromRef,devFromRef))
                        eclDst[clusterIt] = np.linalg.norm(currentSamples[sampleIt] - currentClusters[clusterIt])
                    currentSampleCluster = np.argmin(eclDst) # What cluster the current sample belongs to
                    sampleOwnership[sampleIt] = currentSampleCluster
                currAccDist = accumDist(currentSamples,currentClusters,sampleOwnership)

                if (currAccDist/prevAccDist) > 0.99:
                    break
                prevAccDist = currAccDist

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


def NN(testSamples,templates,templateLabels):
    nTemplates = (np.shape(templates))[0]
    nTests = (np.shape(testSamples))[0]
    eclDst = np.empty(nTemplates)
    match = np.empty(nTests)
    tstAns = np.empty(nTests)
    for testIt in range(nTests):
        for templateIt in range(nTemplates):
            # devFromSample = tstimg[testIt] - templates[templateIt]
            # eclDst[templateIt] = np.sum(np.multiply(devFromSample,devFromSample))
            eclDst[templateIt] = np.linalg.norm(tstimg[testIt] - img[templateIt])
        closestMatch = np.argmin(eclDst)    # Index of closest template
        match[testIt] = closestMatch    # Template index that matched to current test image
        tstAns[testIt] = templateLabels[closestMatch]
    print(eclDst)
    return tstAns, match

def KNN(testSamples,templates,templateLabels,K=7):
    nTemplates = (np.shape(templates))[0]
    nTests = (np.shape(testSamples))[0]
    match = np.empty(nTests)
    tstAns = np.empty(nTests)
    nClasses = int(templateLabels.max())
    eclDst = np.empty(nTemplates)
    kMatches = np.empty(K,dtype=int)  # The K closest templates
    for testIt in range(nTests):
        
        for templateIt in range(nTemplates):
            # devFromSample = tstimg[testIt] - templates[templateIt]
            # eclDst[templateIt] = np.sum(np.multiply(devFromSample,devFromSample))
            eclDst[templateIt] = np.linalg.norm(tstimg[testIt] - img[templateIt])
        
        # Code particular for KNN (vs NN) start

        classFq = np.zeros(nClasses+1)    # Amount of each class in kMatches
        # classAllowed = np.zeros(nClasses+1)   # Whether or not a class has the same frequency as the max frequency

        for kIt in range(K):
            closestMatch = np.argmin(eclDst)    # Index of closest template of the remaining distances
            kMatches[kIt] = closestMatch
            classFq[int(templateLabels[closestMatch])] += 1
            eclDst[closestMatch] = 123456789  # Must be high so sample isnt picked out again by argmin

        maxOccurrences = classFq.max()  # Amount of times most common class has occurred
        # for classIt in range(nClasses): # This is to make sure things go smoothly if more two classes have the max amount of occurrences
        #     if classFq[classIt] == maxOccurrences:
        #         classAllowed[classIt] = 1

        for kIt in range(K):
            if (classFq[int(templateLabels[kMatches[kIt]])] == maxOccurrences):
                bestMatch = kMatches[kIt]   # Matches with 'avaliable' template with lowest distance
                # print('KNN worked, classFq = ', classFq)
                break

        # Code particular for KNN (vs NN) start

        match[testIt] = bestMatch    # Template index that matched to current test image
        tstAns[testIt] = templateLabels[closestMatch]
    return tstAns, match

# Code for saving the cluster data

# clusters, clusterSol = clusters(img,lb,maxClusters=nClusters)

# with open('Data/cluster_images.bin','wb') as binaryFile:
#     binaryFile.write((clusters.astype(np.uint8)).tobytes())

# with open('Data/cluster_labels.bin','wb') as binaryFile:
#     binaryFile.write((clusterSol.astype(np.uint8)).tobytes())


# Code for opening the cluster data

with open('Data/cluster_images.bin','rb') as binaryFile:
    clustersB = binaryFile.read()
with open('Data/cluster_labels.bin','rb') as binaryFile:
    clusterSolB = binaryFile.read()

clusterSol = np.frombuffer(clusterSolB, dtype=np.uint8)
clusters = np.reshape(np.frombuffer(clustersB, dtype=np.uint8), (nClusters*nClasses,784))

# Evaluation

tstAnsandMatch = NN(tstimg,clusters,clusterSol)
NNtstAns = tstAnsandMatch[0]
NNmatch = tstAnsandMatch[1]

tstAnsandMatch = KNN(tstimg,clusters,clusterSol)
KNNtstAns = tstAnsandMatch[0]
KNNmatch = tstAnsandMatch[1]

# print('NN answers:\n',NNtstAns)
# print('KNN answers:\n',NNtstAns)

confMerrR = confMatrix(KNNtstAns,tstlb)
print('Confusion matrix with ',nClusters*nClasses,' references & ',nTests,' tests : \n',confMerrR[0])
print('Error rate : ',confMerrR[1])

answerPlt = np.reshape(tstimg[400],(28,28))
solutionPlt = np.reshape(clusters[int(KNNmatch[400])],(28,28))

plt.suptitle('??? classified number')

plt.subplot(1,2,1)
plt.imshow(answerPlt,cmap='gray',vmin=0,vmax=255)
plt.title('This test image matched')

plt.subplot(1,2,2)
plt.imshow(solutionPlt,cmap='gray',vmin=0,vmax=255)
plt.title('With this reference image')

plt.show()