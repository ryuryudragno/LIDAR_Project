import math
import numpy as np

x_min = 0  # 0
x_max = 1.9  # 1.9
y_min = 0.1  # 0.1
y_max = 3  # 3
z_min = -1
z_max = 1

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


z_min_100 = -1  # -0.25
z_min_101 = -1  # -0.15
z_min_102 = -1  # -0.35
z_min_103 = -1  # -0.35

outlier = -100
unit_matrix = np.eye(3)

# 26度回転
trans_init_100z = np.asarray(
    [
        [2 / math.sqrt(5), -1 / math.sqrt(5), 0],
        [1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# -26度回転
trans_init_101z = np.asarray(
    [
        [2 / math.sqrt(5), 1 / math.sqrt(5), 0],
        [-1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# 37度回転→101と180回転差に
trans_init_102z = np.asarray(
    [
        [-0.80, 0.60, 0],
        [-0.60, -0.80, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# -37度回転+180度回転
trans_init_103z = np.asarray(
    [
        [-0.80, -0.60, 0],
        [0.60, -0.80, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# z軸周り回転
# 5度くらい
trans_init_100x = np.asarray(
    [
        [1, 0, 0],
        [0, 0.996, -0.0879],
        [0.0, 0.0879, 0.996],
    ]
)
# -1度回転
trans_init_101x = np.asarray(
    [
        [1, 0, 0],
        [0, 0.9998, 0.0175],
        [0.0, -0.0175, 0.9998],
    ]
)
# -5度回転
trans_init_102x = np.asarray(
    [
        [1, 0, 0],
        [0, 0.996, 0.0879],
        [0.0, -0.0879, 0.996],
    ]
)

trans_init_103x = np.asarray(
    [
        [1, 0, 0],
        [0, 0.9994, 0.0349],
        [0.0, -0.0349, 0.9994],
    ]
)

zmin = [z_min_100, z_min_101, z_min_102, z_min_103]
transarray_z = [trans_init_100z, trans_init_101z, trans_init_102z, trans_init_103z]
transarray_x = [trans_init_100x, trans_init_101x, trans_init_102x, trans_init_103x]
# print(transarray)
# print(transarray[0])
# print(trans_init_100)

trans_carib1 = np.eye(4)
trans_carib2 = np.asarray(
    [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)
trans_carib = []
