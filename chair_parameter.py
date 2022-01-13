import math
import numpy as np

x_min = 0  # 0
x_max = 1.9  # 1.9
y_min = 0.1  # 0.1
y_max = 3  # 3

# 壁あり
# x_min = -0.5  # 0
# x_max = 2.5  # 1.9
# y_min = -0.5  # 0.1
# y_max = 4.0  # 3


x_min_100 = -2.5
x_max_100 = 0.5
x_min_101 = 0
x_max_101 = 2
x_min_102 = 0
x_max_102 = 2
x_min_103 = -2.5
x_max_103 = 0.5

# y_min = -3.5  # or 3.5
z_min_100 = z_min_102 = -0.35
z_min_101 = -0.15

outlier = -100

# 26度回転
trans_init_100 = np.asarray(
    [
        [2 / math.sqrt(5), -1 / math.sqrt(5), 0],
        [1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
print(type(trans_init_100))
# -26度回転
trans_init_101 = np.asarray(
    [
        [2 / math.sqrt(5), 1 / math.sqrt(5), 0],
        [-1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# 37度回転→101と180回転差に
trans_init_102 = np.asarray(
    [
        [-0.80, 0.60, 0],
        [-0.60, -0.80, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# -37度回転+180度回転
trans_init_103 = np.asarray(
    [
        [-0.80, -0.60, 0],
        [0.60, -0.80, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

transarray = [trans_init_100, trans_init_101, trans_init_102, trans_init_103]
# print(transarray)
# print(transarray[0])
# print(trans_init_100)
