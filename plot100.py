import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import open3d as o3
import math
import chair_parameter as param


def calc_distance(inputFrame):
    return (inputFrame["X"] ** 2 + inputFrame["Y"] ** 2 + inputFrame["Z"] ** 2) ** 0.5


targetData = pd.read_csv("datasets_lidar/chair/chair_1921680100.csv")
# targetData = pd.read_csv("datasets_lidar/boxPosition1/boxPosition1_1921680101.csv")
# rubishBinWithBricks
# twoBricks
# boxPosition1
# chair

## remove outliers which are further away then 10 meters
dist_targetData = calc_distance(targetData)

dist_threshold = 10  # 閾値
targetData = targetData.iloc[
    np.nonzero((dist_targetData < dist_threshold).values)[0], :
]
targetData = targetData.iloc[np.nonzero((targetData["Z"] > -0.35).values)[0], :]
targetData = targetData.iloc[np.nonzero((targetData["Z"] < 0.8).values)[0], :]
targetData = targetData.iloc[np.nonzero((targetData["Y"] < 6.5).values)[0], :]

# targetData = targetData.T
# target data
targetMatrix = np.array([targetData["X"], targetData["Y"], targetData["Z"]])
# print(type(targetMatrix))
# print((targetMatrix))


trans_init = np.asarray(
    [
        [2 / math.sqrt(5), -1 / math.sqrt(5), 0],
        [1 / math.sqrt(5), 2 / math.sqrt(5), 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# rotate
targetMatrix = np.dot(param.trans_init_100, targetMatrix)

# 整理
targetMatrix = np.where(
    (targetMatrix[0] > param.x_min_100) & (targetMatrix[0] < param.x_max_100),
    targetMatrix,
    param.outlier,
)
targetMatrix = np.where((targetMatrix[1] < param.y_max), targetMatrix, param.outlier)


targetMatrix = targetMatrix.T
targetMatrix = targetMatrix[np.all(targetMatrix != param.outlier, axis=1), :]
print(targetMatrix)
print(len(targetMatrix))

# 3D散布図でプロットするデータを生成する為にnumpyを使用
X = targetMatrix[:, 0]  # 自然数の配列
Y = targetMatrix[:, 1]  # 特に意味のない正弦
Z = targetMatrix[:, 2]  # 特に意味のない正弦
# print(X)

# グラフの枠を作成
fig = plt.figure()
ax = Axes3D(fig)

# X,Y,Z軸にラベルを設定
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# .plotで描画
ax.plot(X, Y, Z, marker="o", linestyle="None")

# 最後に.show()を書いてグラフ表示
plt.show()
