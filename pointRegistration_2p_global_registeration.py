from datetime import time
import pandas as pd
import open3d as o3
import numpy as np
import copy
import math

# import euler


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
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556],
    )


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


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    print("Loading files")
    # sourceData = pd.read_csv("datasets_lidar/chair/chair_1921680100.csv")
    # targetData = pd.read_csv("datasets_lidar/chair/chair_1921680101.csv")
    sourceData = pd.read_csv("datasets_lidar/boxPosition1/boxPosition1_1921680101.csv")
    targetData = pd.read_csv("datasets_lidar/boxPosition1/boxPosition1_1921680102.csv")

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

    # Rotate around the x-axis
    # trans_init = np.asarray(
    #     [
    #         [1, 0, 0],
    #         [0, 1 / 2, -math.sqrt(3) / 2],
    #         [0.0, math.sqrt(3) / 2, 1 / 2],
    #     ]
    # )
    # Rotate around the y-axis
    trans_init = np.asarray(
        [
            [math.sqrt(2) / 2, 0, math.sqrt(2) / 2],
            [0.0, 1, 0.0],
            [-math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2],
        ]
    )
    # # Rotate around the z-axis
    # trans_init = np.asarray(
    #     [
    #         [1 / 2, -math.sqrt(3) / 2, 0],
    #         [math.sqrt(3) / 2, 1 / 2, 0.0],
    #         [0.0, 0.0, 1.0],
    #     ]
    # )

    # y軸周りに45度
    # trans_init = np.asarray(
    #     [
    #         [1 / 2, 0.0, math.sqrt(3) / 2, 0.0],
    #         [0.0, 1.0, 0.0, 0.0],
    #         [-math.sqrt(3) / 2, 0.0, 1 / 2, 0.0],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )

    # source data
    print("Transforming source data")
    source = o3.geometry.PointCloud()  # 0pointの生成

    sourceMatrix = np.array([sourceData["X"], sourceData["Y"], sourceData["Z"]])
    print(sourceMatrix)
    sourceMatrix = np.dot(trans_init, sourceMatrix)
    sourceMatrix = sourceMatrix.transpose()
    print("a")
    print(sourceMatrix)
    source.points = o3.utility.Vector3dVector(sourceMatrix)
    # print(source.points)

    # target data
    print("Transforming target data")
    target = o3.geometry.PointCloud()
    targetMatrix = np.array(
        [targetData["X"], targetData["Y"], targetData["Z"]]
    ).transpose()
    target.points = o3.utility.Vector3dVector(targetMatrix)

    draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 0.5
    print(
        ":: Apply fast global registration with distance threshold %.3f"
        % distance_threshold
    )
    result = o3.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result_ransac.transformation,
        o3.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


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

    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    # print(result_ransac)
    # print(source)
    # print(target)
    # print(source_down)
    # print(target_down)
    # print(source_fpfh)
    # print(target_fpfh)

    # this does not work yet - error
    #
    # result_icp = refine_registration(
    #     source, target, source_fpfh, target_fpfh, voxel_size
    # )
    # print(result_icp)
    # draw_registration_result(source, target, result_icp.transformation)
