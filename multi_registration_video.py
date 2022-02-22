import statistics
from time import time
from matplotlib import animation
from numpy.lib.utils import source
import pandas as pd
import open3d as o3
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyautogui
import time

from sklearn import datasets

import chair_parameter as param
import drafts.read_video_csv as vd

# 複数点
# http://www.open3d.org/docs/0.13.0/tutorial/pipelines/multiway_registration.html
# Global registartion
# http://www.open3d.org/docs/0.13.0/tutorial/pipelines/global_registration.html


def calc_distance(inputFrame):
    return (inputFrame["X"] ** 2 + inputFrame["Y"] ** 2 + inputFrame["Z"] ** 2) ** 0.5


def distance_cut(sourceData, dist_data, dist_threshold):
    distance_cut = sourceData.iloc[
        np.nonzero((dist_data < dist_threshold).values)[0], :
    ]

    return distance_cut


# rotation and remove unnecessary data
def source_preprocess(
    sourceData,
    trans_init_z,
    trans_init_x,
    trans_init_y,
    x_min,
    x_max,
    y_min,
    y_max,
    z_max,
    z_min,
    z_adjust,
    trans_carib,
    outlier,
):
    source = o3.geometry.PointCloud()  # generate point_cloud
    sourceMatrix = np.array([sourceData["X"], sourceData["Y"], sourceData["Z"]])

    # ここから回転 and cut
    sourceMatrix = np.dot(trans_init_z, sourceMatrix)
    sourceMatrix = np.dot(trans_init_x, sourceMatrix)
    sourceMatrix = np.dot(trans_init_y, sourceMatrix)

    medX = statistics.median(sourceMatrix[0])
    medY = statistics.median(sourceMatrix[1])
    medZ = statistics.median(sourceMatrix[2])

    if medX < 0:
        sourceMatrix[0] = sourceMatrix[0] + 2
    if medY < 0:
        sourceMatrix[1] = sourceMatrix[1] + 3.5
    # if medZ < 0.1:
    #     sourceMatrix[2] = sourceMatrix[2] + 0.2
    sourceMatrix[2] = sourceMatrix[2] + z_adjust

    sourceMatrix = np.where(
        (sourceMatrix[0] > x_min) & (sourceMatrix[0] < x_max), sourceMatrix, outlier
    )
    sourceMatrix = np.where(
        (sourceMatrix[1] > y_min) & (sourceMatrix[1] < y_max), sourceMatrix, outlier
    )
    sourceMatrix = np.where(
        (sourceMatrix[2] > z_min) & (sourceMatrix[2] < z_max), sourceMatrix, outlier
    )
    # axisで行か列かを指定できる
    sourceMatrix = sourceMatrix[:, np.all(sourceMatrix != outlier, axis=0)]

    sourceMatrix = sourceMatrix.T

    source.points = o3.utility.Vector3dVector(sourceMatrix)
    # source.transform(trans_carib)

    return source


def prepare_dataset(voxel_size, num, time_arr):
    print(":: Load two point clouds and disturb initial pose.")
    print("Loading files")

    # sources = pcds = np.zeros(((50, 4)), dtype=o3.cpu.pybind.geometry.PointCloud)
    sources = []  # 初期化は分けないと同じものを指すことになる
    pcds = []
    # print(len(time_arr))
    dist_threshold = 10  # 距離の閾値(原点から遠すぎるものを排除？)
    for m in range(len(time_arr)):
        source_time = []
        pcd_time = []
        for i in range(4):
            # read the data
            sourceData = pd.read_csv(
                vd.readData_multi[num] % str(i) + time_arr[m] + ".csv"
            )

            # remove outliers which are further away then 10 meters
            dist_sourceData = calc_distance(sourceData)  # 原点からの距離を計算

            # print(sourceData(dist_sourceData>20))
            sourceData = distance_cut(sourceData, dist_sourceData, dist_threshold)
            # print(sourceData)

            # source data
            # print("Transforming source data")
            source = source_preprocess(
                sourceData,
                param.transarray_z[i],
                param.transarray_x[i],
                param.transarray_y[i],
                param.x_min,
                param.x_max,
                param.y_min,
                param.y_max,
                param.z_max,
                param.z_min[i],
                param.z_adjust[i],
                param.trans_carib[i],
                param.outlier,
            )

            # print(":: Downsample with a voxel size %.3f." % voxel_size)
            pcd_down = source.voxel_down_sample(voxel_size)

            ##この4行入れたらicp_reguration_errorが出なくなった？
            radius_normal = voxel_size * 2
            # print(":: Estimate normal with search radius %.3f." % radius_normal)
            pcd_down.estimate_normals(
                o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )

            source_time.append(source)
            pcd_time.append(pcd_down)
            # pcds[i][m] = pcd_down
        # print(source_time)
        # print(pcd_time)
        sources.append(source_time)
        # print(len(sources))
        pcds.append(pcd_time)
    # print(sources)
    # print(le(pcds))
    return sources, pcds


def get_time(path):
    files = os.listdir(path)
    files.sort()
    files = files[:timestep]
    timestamp_arr = []
    for word in files:
        a = word.rsplit("_", 1)
        b = a[1].split(".", 1)
        c = b[0]
        # c = int(c)
        timestamp_arr.append(c)
        # print(type(c))
    # print(timestamp_arr)
    # print(len(timestamp_arr))
    return timestamp_arr


def plot(pcds_array, i, outlier):
    # グラフの枠を作成
    fig = plt.figure()
    ax = Axes3D(fig)

    # X,Y,Z軸にラベルを設定
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    col = np.arange(30)
    # pcds_array[i] = np.where((pcds_array[i][2] < 0), pcds_array[i], outlier)
    # # axisで行か列かを指定できる
    # pcds_array[i] = pcds_array[i][:, np.all(pcds_array[i] != outlier, axis=0)]

    # X,Y,Z軸にラベルを設定
    ax.set_xlim3d(-0.2, 1.8)
    ax.set_ylim3d(-0.2, 3.0)
    ax.set_zlim3d(-0.4, 0.8)
    ax.view_init(elev=30, azim=45)
    # .plotで描画
    ax.scatter(
        pcds_array[i][0],
        pcds_array[i][1],
        pcds_array[i][2],
        marker="o",
        linestyle="None",
        c=[
            1 if pcds_array[i][2][j] < -0.15 else 0
            for j in range(len(pcds_array[i][2]))
        ],
        # cmap="ocean",
    )
    # print(pcds_array[i][2])

    # # 最後に.show()を書いてグラフ表示
    plt.show()
    time.sleep(0.05)
    pyautogui.press(["q"])


if __name__ == "__main__":
    agentNum = 4
    timestep = 50
    # 0→spinningCrane,1→oscillating
    n = 1
    path = vd.read_timestep[n]
    time_arr = get_time(path)

    voxel_size = 0.05  # means 5cm for the dataset

    sources, pcds_down = prepare_dataset(voxel_size, n, time_arr)
    print(pcds_down)

    # # 7
    pcd_combined_down = any

    print("Transform points and display")
    pcds_array = []
    i = 0
    for pcd in pcds_down:  # 50回
        pcd_combined = o3.geometry.PointCloud()
        # print(pcd)
        for point_id in range(len(pcd)):  # 4回
            # print(pose_graph.edges[point_id])
            pcd[point_id].transform(param.trans_carib[point_id])
            pcd_combined += pcd[point_id]

        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
        pcd_array = []
        X = []
        Y = []
        Z = []
        # print(len(pcds_down))
        for x in pcd_combined_down.points:
            # pcd_array.append(x)
            X.append(x[0])
            Y.append(x[1])
            Z.append(x[2])
        pcd_array = [X, Y, Z]
        # print(len(pcd_array))
        pcds_array.append(pcd_array)

    # print(len(pcds_array))
    # print(len(pcds_array[0]))

    # print(type(pcd_combined_down))
    # print(type(pcd_combined_down.points))
    # print(pcds_array[49][0] == X)

    # plot each time(First press q, then it works automatically)
    for index in range(len(pcds_array)):
        plot(pcds_array, index, param.outlier)

    # # ここからアニメ it doen't work
    # def func(num, dataSet, scatters):
    #     # NOTE: there is no .set_data() for 3 dim data...
    #     # print(line)
    #     for i in range(dataSet[0].shape[0]):
    #         scatters[i]._offsets3d = (
    #             dataSet[num][0:1],
    #             dataSet[num][1:2],
    #             dataSet[num][2:],
    #         )
    #     return scatters
    #     # a = np.array(dataSet[num][2], dtype=object)
    #     # print(type(a))
    #     # line.set_3d_properties(a)

    #     # print(type(dataSet[num][0:2]))
    #     # print(type(np.array(dataSet[num][2])))
    #     return line

    # # THE DATA POINTS
    # dataSet = np.array(pcds_array, dtype=object)
    # # print(type(dataSet))
    # numDataPoints = len(pcds_array)

    # # GET SOME MATPLOTLIB OBJECTS
    # fig = plt.figure()
    # ax = Axes3D(fig)

    # # NOTE: Can't pass empty arrays into 3d version of plot()
    # scatters = [
    #     ax.scatter(dataSet[0][0:1], dataSet[0][1:2], dataSet[0][2:], lw=2, c="g")
    # ]
    # # For line plot
    # # print(plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c="g"))
    # # print(plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c="g")[0])

    # # AXES PROPERTIES]
    # # ax.set_xlim3d([limit0, limit1])
    # ax.set_xlabel("X(t)")
    # ax.set_ylabel("Y(t)")
    # ax.set_zlabel("time")
    # ax.set_title("Trajectory of electron for E vector along [120]")

    # # Creating the Animation object
    # line_ani = animation.FuncAnimation(
    #     fig,
    #     func,
    #     frames=numDataPoints,
    #     fargs=(dataSet, scatters),
    #     interval=50,
    #     blit=False,
    # )

    # # print(dataSet[0])
    # # func(3, dataSet, scatters)
    # # line_ani.save("AnimationNew.mp4")
    # Writer = animation.writers["ffmpeg"]
    # writer = Writer(
    #     fps=30,
    #     metadata=dict(artist="Me"),
    #     bitrate=1800,
    #     extra_args=["-vcodec", "libx264"],
    # )
    # line_ani.save("3d-scatted-animated.mp4", writer=writer)

    # # plt.show()
