import numpy as np
import matplotlib.pyplot as plt

nSamples = 6000
nTests = 500
nClasses = 10
nClusters = 64
ourK = 7

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

def accumDist(samples,clusters,sampleOwnership):
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
    clusters = 500 * np.ones((maxClusters*nClasses,784))
    clusterSol = np.empty(maxClusters*nClasses)
    for classIt in range(nClasses):
        for clusterIt in range(maxClusters):
            clusterSol[maxClusters*classIt + clusterIt] = classIt
    sortedImg = sortedImage(img,lb)
    _, classCount = np.unique(lb, return_counts = True)
    startIndex = 0

    for classIt in range(nClasses):
        classLength = classCount[classIt] 
        print('Current class is : ', classIt, ' With ', classLength, ' samples of the current class')
        currentSamples = sortedImg[startIndex:startIndex+classLength]
        currentClusters = clusters[maxClusters*classIt:maxClusters*classIt+maxClusters]
        
        # Alg from compendium start
        for sampleIt in range(classLength):
            currentClusters[0] += currentSamples[sampleIt]
        currentClusters[0] = currentClusters[0] / classLength
        for amountOfClustersIt in range(1,maxClusters+1):
            prevAccDist = 123456789
            mostPopularCluster = 0
            while True:
                sampleOwnership = np.empty(classLength,dtype=np.int)
                eclDst = np.empty(maxClusters)
                for sampleIt in range(classLength):
                    for clusterIt in range(maxClusters):
                        eclDst[clusterIt] = np.linalg.norm(currentSamples[sampleIt] - currentClusters[clusterIt])
                    currentSampleCluster = np.argmin(eclDst)
                    sampleOwnership[sampleIt] = currentSampleCluster
                currAccDist = accumDist(currentSamples,currentClusters,sampleOwnership)
                if (currAccDist/prevAccDist) > 0.99:
                    break
                prevAccDist = currAccDist
                maxHits = 0
                for clusterIt in range(amountOfClustersIt):
                    currentClusters[clusterIt] = np.zeros(784)
                    hits = 0
                    for sampleIt in range(classLength):
                        if sampleOwnership[sampleIt] == clusterIt:
                            currentClusters[clusterIt] += currentSamples[sampleIt]
                            hits += 1
                            if hits > maxHits:
                                maxHits = hits
                                mostPopularCluster = clusterIt
                    currentClusters[clusterIt] = currentClusters[clusterIt] / hits
            if amountOfClustersIt < maxClusters:
                currentClusters[amountOfClustersIt] = currentClusters[mostPopularCluster] + (0.2*np.random.rand(784)-0.1)
        for clusterIt in range(maxClusters):
            clusters[maxClusters*classIt+clusterIt] = currentClusters[clusterIt]
        # Alg from compendium end

        startIndex += classLength
    return clusters, clusterSol

def sortedImage(img,lb):
    dataDims = np.shape(img)
    nSamples = dataDims[0]
    solDims = np.shape(lb)
    nClasses = solDims[0]
    sortedImg = np.empty((nSamples,784))
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
            eclDst[templateIt] = np.linalg.norm(tstimg[testIt] - img[templateIt])
        closestMatch = np.argmin(eclDst)
        match[testIt] = closestMatch
        tstAns[testIt] = templateLabels[closestMatch]
    return tstAns, match

def KNN(testSamples,templates,templateLabels,K=7):
    nTemplates = (np.shape(templates))[0]
    nTests = (np.shape(testSamples))[0]
    match = np.empty(nTests)
    tstAns = np.empty(nTests)
    nClasses = int(templateLabels.max())
    eclDst = np.empty(nTemplates)
    kMatches = np.empty(K,dtype=int)
    for testIt in range(nTests):
        for templateIt in range(nTemplates):
            eclDst[templateIt] = np.linalg.norm(tstimg[testIt] - img[templateIt])

        # Code particular for KNN (vs NN) start
        classFq = np.zeros(nClasses+1)
        for kIt in range(K):
            closestMatch = np.argmin(eclDst)
            kMatches[kIt] = closestMatch
            classFq[int(templateLabels[closestMatch])] += 1
            eclDst[closestMatch] = 123456789
        maxOccurrences = classFq.max()
        for kIt in range(K):
            if (classFq[int(templateLabels[kMatches[kIt]])] == maxOccurrences):
                bestMatch = kMatches[kIt]
                break
        # Code particular for KNN (vs NN) end

        match[testIt] = bestMatch
        tstAns[testIt] = templateLabels[closestMatch]
    return tstAns, match

# Code for saving the cluster data (can be commented out once clustering is complete)

clusters, clusterSol = clusters(img,lb,maxClusters=nClusters)
with open('Data/cluster_images.bin','wb') as binaryFile:
    binaryFile.write((clusters.astype(np.uint8)).tobytes())
with open('Data/cluster_labels.bin','wb') as binaryFile:
    binaryFile.write((clusterSol.astype(np.uint8)).tobytes())

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
tstAnsandMatch = KNN(tstimg,clusters,clusterSol,K=ourK)
KNNtstAns = tstAnsandMatch[0]
KNNmatch = tstAnsandMatch[1]

NNconfMerrR = confMatrix(NNtstAns,tstlb)
print('NN Confusion matrix with ',nClusters*nClasses,' references & ',nTests,' tests : \n',NNconfMerrR[0])
print('Error rate : ',NNconfMerrR[1])
KNNconfMerrR = confMatrix(KNNtstAns,tstlb)
print('KNN confusion matrix with ',nClusters*nClasses,' references & ',nTests,' tests : \n',KNNconfMerrR[0])
print('Error rate : ',KNNconfMerrR[1])

answerPlt = np.reshape(tstimg[0],(28,28))
solutionPlt = np.reshape(clusters[int(NNmatch[0])],(28,28))
plt.subplot(1,2,1)
plt.imshow(answerPlt,cmap='gray',vmin=0,vmax=255)
plt.title('This test image matched')
plt.subplot(1,2,2)
plt.imshow(solutionPlt,cmap='gray',vmin=0,vmax=255)
plt.title('With this reference')
plt.show()
