import math
import open3d as o3
import numpy as np
import copy
import pandas as pd
import os


def calc_distance(inputFrame):
    return (inputFrame["X"] ** 2 + inputFrame["Y"] ** 2 + inputFrame["Z"] ** 2) ** 0.5


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.6459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    print("Loading files")
    sourceData = pd.read_csv("datasets_lidar/chair/chair_1921680100.csv")
    targetData = pd.read_csv("datasets_lidar/chair/chair_1921680102.csv")

    ## remove outliers which are further away then 10 meters
    dist_sourceData = calc_distance(sourceData)
    dist_targetData = calc_distance(targetData)

    dist_threshold = 10  # 閾値
    sourceData = sourceData.iloc[
        np.nonzero((dist_sourceData < dist_threshold).values)[0], :
    ]
    targetData = targetData.iloc[
        np.nonzero((dist_targetData < dist_threshold).values)[0], :
    ]

    # source data
    print("Transforming source data")
    source = o3.geometry.PointCloud()  # 0pointの生成

    sourceMatrix = np.array(
        [sourceData["X"], sourceData["Y"], sourceData["Z"]]
    ).transpose()
    source.points = o3.utility.Vector3dVector(sourceMatrix)

    # target data
    print("Transforming target data")
    target = o3.geometry.PointCloud()
    targetMatrix = np.array(
        [targetData["X"], targetData["Y"], targetData["Z"]]
    ).transpose()
    target.points = o3.utility.Vector3dVector(targetMatrix)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


if __name__ == "__main__":
    voxel_size = 0.05  # means 5cm for the dataset
    (
        source,
        target,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ) = prepare_dataset(voxel_size)
    threshold = 0.02
    trans_init = np.asarray(
        [
            [0.0, 0, 1.0, 0.0],
            [0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    draw_registration_result(source, target, trans_init)

    # 4
    print("Initial alignment")
    evaluation = o3.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init
    )
    print(evaluation)

    # 5
    print("Apply point-to-point ICP")
    reg_p2p = o3.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)
