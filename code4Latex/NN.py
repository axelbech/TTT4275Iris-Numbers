import numpy as np
import matplotlib.pyplot as plt

nSamples = 1000
nTests = 200
nClasses = 10

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

eclDst = np.empty(nSamples)
match = np.empty(nTests)
tstAns = np.empty(nTests)
for testIt in range(nTests):
    for sampleIt in range(nSamples):
        eclDst[sampleIt] = np.linalg.norm(tstimg[testIt] - img[sampleIt])
    closestMatch = np.argmin(eclDst)
    match[testIt] = closestMatch
    tstAns[testIt] = lb[closestMatch]

confMerrR = confMatrix(tstAns,tstlb)
print('Confusion matrix with ',nSamples,' training samples & ',nTests,' tests : \n',confMerrR[0])
print('Error rate : ',confMerrR[1])

answerPlt = np.reshape(tstimg[0],(28,28))
solutionPlt = np.reshape(img[int(match[0])],(28,28))
plt.subplot(1,2,1)
plt.imshow(answerPlt,cmap='gray',vmin=0,vmax=255)
plt.title('This test image matched')
plt.subplot(1,2,2)
plt.imshow(solutionPlt,cmap='gray',vmin=0,vmax=255)
plt.title('With this training image')
plt.show()