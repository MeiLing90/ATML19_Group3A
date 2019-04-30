#%% Testing out plots generation
import matplotlib.pyplot as plt
import numpy as np
for i in range(100):
    b = np.array([[i*1, 255, 233], [1, i*2, 233], [1, 255, i*3]])
    plt.imshow(b)
    plt.show()

#%% Testing out something else
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([9, 1, 2, 1, 2, 1, 7, 8, 9])
plt.plot(x, y)
plt.show()
