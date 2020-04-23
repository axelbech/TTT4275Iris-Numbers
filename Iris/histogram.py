# start

import numpy as np
import matplotlib.pyplot as plt

testArray = [1, 4, 9, 26, 25, 36, 49, 64, 81, 100, 10, 20, 60, 70, 80, 90, 100]
bins = [0, 33, 66, 100]

plt.hist(testArray, bins)
plt.show()