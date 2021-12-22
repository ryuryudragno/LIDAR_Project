import math
import numpy as np


x_min_100 = -2
x_max_100 = 0
x_min_101 = 0
x_max_101 = 2
y_max = 3.5
outlier = -100

trans_init_100 = np.asarray(
    [
        [2 / math.sqrt(5), -1 / math.sqrt(5), 0],
        [1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)

trans_init_101 = np.asarray(
    [
        [2 / math.sqrt(5), 1 / math.sqrt(5), 0],
        [-1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
