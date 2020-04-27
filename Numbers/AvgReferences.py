import numpy as np
import matplotlib.pyplot as plt

nSamples = 1000
nTests = 1000
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

print('Our answers are : \n', tstAns[0:24])
print('The real answers are : \n', tstlb[0:24])

plt.imshow(np.reshape(tstimg[11],(28,28)),cmap='gray',vmin=0,vmax=255)

plt.show()

ref0 = np.reshape(ref[0],(28,28))
ref1 = np.reshape(ref[1],(28,28))
ref2 = np.reshape(ref[2],(28,28))
ref3 = np.reshape(ref[3],(28,28))
ref4 = np.reshape(ref[4],(28,28))
ref5 = np.reshape(ref[5],(28,28))
ref6 = np.reshape(ref[6],(28,28))
ref7 = np.reshape(ref[7],(28,28))
ref8 = np.reshape(ref[8],(28,28))
ref9 = np.reshape(ref[9],(28,28))

plt.subplot(5,2,1)
plt.imshow(ref0,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,2)
plt.imshow(ref1,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,3)
plt.imshow(ref2,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,4)
plt.imshow(ref3,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,5)
plt.imshow(ref4,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,6)
plt.imshow(ref5,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,7)
plt.imshow(ref6,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,8)
plt.imshow(ref7,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,9)
plt.imshow(ref8,cmap='gray',vmin=0,vmax=255)

plt.subplot(5,2,10)
plt.imshow(ref9,cmap='gray',vmin=0,vmax=255)

plt.show()