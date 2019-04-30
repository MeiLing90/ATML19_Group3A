import matplotlib.pyplot as plt
import numpy as np
for i in range(100):
    b = np.array([[i*1, 255, 233], [1, i*2, 233], [1, 255, i*3]])
    plt.imshow(b)
    plt.show()