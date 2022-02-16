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

# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" :
# 	[
# 		{
# 			"boundingbox_max" : [ 1.8787309052183363, 2.809943272756724, 0.38713398751026762 ],
# 			"boundingbox_min" : [ -0.013912791113105644, 0.49133782330522496, -0.28526034726109101 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.77920625806744848, -0.57987940671965754, 0.23786021325766751 ],
# 			"lookat" : [ -0.12481240763005762, 1.0039607063096871, 0.81194244371714996 ],
# 			"up" : [ 0.19116025605760037, 0.1415469754318206, 0.97129923826290332 ],
# 			"zoom" : 0.5158999999999998
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }
