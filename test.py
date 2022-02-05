import numpy as np

a = np.arange(12).reshape((3, 4))

# print(a[:, np.any(a < 5, axis=0)])
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

b = a[np.any(a < 5, axis=1)]
# print(b)

# print(a[0])
c = np.where((a[0] > 0) & (a[0] < 2), a, 0)

t = np.arange(0, 20, 1)  # This would be the z-axis ('t' means time here)
x = np.cos(t) - 1
y = 1 / 2 * (np.cos(2 * t) - 1)
dataSet = np.array([x, y, t])
print(dataSet[0])
print(dataSet[0:2, :2])
