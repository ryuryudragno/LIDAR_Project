import math
import numpy as np
import pandas as pd

x_min = 0  # 0
x_max = 1.9  # 1.9
y_min = 0.1  # 0.1
y_max = 3  # 3
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
z_min = [z_min_100, z_min_101, z_min_102, z_min_103]

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
# 2度回転
trans_init_103x = np.asarray(
    [
        [1, 0, 0],
        [0, 0.9986, 0.0523],
        [0.0, -0.0523, 0.9986],
    ]
)

# 度回転
trans_init_103y = np.asarray(
    [
        [0.9986, 0, 0.0523],
        [0, 1, 0],
        [-0.0523, 0.0, 0.9986],
    ]
)
trans_init_100y = trans_init_101y = trans_init_102y = np.eye(3)

transarray_z = [trans_init_100z, trans_init_101z, trans_init_102z, trans_init_103z]
transarray_x = [trans_init_100x, trans_init_101x, trans_init_102x, trans_init_103x]
transarray_y = [trans_init_100y, trans_init_101y, trans_init_102y, trans_init_103y]
# print(transarray)
# print(transarray[0])
# print(trans_init_100)

trans_carib1 = np.eye(4)
# trans_carib2 = np.asarray(
#     [
#         [0.999, 0.0393, 0.0112, -0.0771],
#         [-0.0394, 0, 0, 0],
#         [1, 0, 0, 0],
#         [1, 0, 0, 0],
#     ]
# )
trans_carib2 = np.asarray(
    [
        [0.99924652, 0.03723041, 0.01096773, -0.07560901],
        [-0.03736168, 0.99922956, 0.01201658, 0.15315943],
        [-0.0105119, -0.0124173, 0.99986765, -0.19562897],
        [0, 0, 0, 1],
    ]
)
trans_carib3 = np.asarray(
    [
        [9.96860129e-01, -2.31209120e-02, -7.57318026e-02, 1.26209613e-02],
        [2.29850218e-02, 9.99732256e-01, -2.66558671e-03, 3.48578065e-02],
        [7.57731567e-02, 9.16519981e-04, 9.97124661e-01, -2.09826850e-02],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
trans_carib4 = np.asarray(
    [
        [0.99475009, -0.10072929, 0.01805204, 0.32094569],
        [0.10064532, 0.99490715, 0.0055037, -0.1096279],
        [-0.01851449, -0.00365795, 0.9998219, 0.06686633],
        [0, 0, 0, 1],
    ]
)


trans_carib = [trans_carib1, trans_carib2, trans_carib3, trans_carib4]


readData_multi = [
    "datasets_lidar/boxBinBrickets/boxBinBrickets_192168010%s.csv",
    "datasets_lidar/boxPosition1/boxPosition1_192168010%s.csv",
    "datasets_lidar/boxPosition2/boxPosition2_192168010%s.csv",
    "datasets_lidar/chair/chair_192168010%s.csv",
    "datasets_lidar/crane/crane_192168010%s.csv",
    "datasets_lidar/rubishBin/rubishBin_192168010%s.csv",
    "datasets_lidar/rubishBinWithBricks/rubishBinWithBricks_192168010%s.csv",
    "datasets_lidar/twoBricks/twoBricks_192168010%s.csv",
]

# 最初に上手くいったやつ
# trans_carib2 = np.asarray(
#     [
#         [9.99163125e-01, 3.93122295e-02, 1.12959418e-02, -7.71988079e-02],
#         [-3.94155018e-02, 9.99181742e-01, 9.06998369e-03, 1.55506416e-01],
#         [-1.09301375e-02, -9.50762846e-03, 9.99895063e-01, -3.97856370e-04],
#         [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
#     ]
# )
# trans_carib3 = np.asarray(
#     [
#         [0.99639894, -0.0410268, -0.07420213, 0.04774881],
#         [0.04052159, 0.99914417, -0.00830186, 0.01895619],
#         [0.07447922, 0.00526518, 0.99720867, -0.02751265],
#         [0, 0, 0, 1],
#     ]
# )
# trans_carib4 = np.asarray(
#     [
#         [0.99077428, -0.11546527, 0.07095144, 0.35902327],
#         [0.11392439, 0.9931645, 0.02540688, -0.11939477],
#         [-0.07340006, -0.01708939, 0.99715615, 0.22807331],
#         [0, 0, 0, 1],
#     ]
# )
