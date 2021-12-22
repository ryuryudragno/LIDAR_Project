from datetime import time
import pandas as pd
import open3d as o3
import numpy as np
import copy

import chair_parameter as param

# import euler
# command qで解除


def calc_distance(inputFrame):
    return (inputFrame["X"] ** 2 + inputFrame["Y"] ** 2 + inputFrame["Z"] ** 2) ** 0.5


def z_cut(sourceData, dist_data, dist_threshold, z_threshold):
    distance_cut = sourceData.iloc[
        np.nonzero((dist_data < dist_threshold).values)[0], :
    ]
    preprocessed_data = distance_cut.iloc[
        np.nonzero((distance_cut["Z"] > z_threshold).values)[0], :
    ]

    return preprocessed_data


def source_preprocess(sourceData, trans_init, x_min, x_max, y_max, outlier):
    source = o3.geometry.PointCloud()  # generate point_cloud
    sourceMatrix = np.array([sourceData["X"], sourceData["Y"], sourceData["Z"]])
    sourceMatrix = np.dot(trans_init, sourceMatrix)
    sourceMatrix = np.where(
        (sourceMatrix[0] > x_min) & (sourceMatrix[0] < x_max), sourceMatrix, outlier
    )
    sourceMatrix = np.where((sourceMatrix[1] < y_max), sourceMatrix, outlier)
    sourceMatrix = sourceMatrix.T
    sourceMatrix = sourceMatrix[np.all(sourceMatrix != outlier, axis=1), :]
    source.points = o3.utility.Vector3dVector(sourceMatrix)

    return source


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.506, 0])  # 100 is red
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 101 is blue
    source_temp.transform(transformation)
    o3.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.6559,  # lager number, smaller object
        front=[-0.6452, 0.5036, 0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.5482, 0.1556],
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
    sourceData = pd.read_csv("datasets_lidar/chair/chair_1921680100.csv")
    targetData = pd.read_csv("datasets_lidar/chair/chair_1921680101.csv")
    # sourceData = pd.read_csv("datasets_lidar/boxPosition1/boxPosition1_1921680100.csv")
    # targetData = pd.read_csv("datasets_lidar/boxPosition1/boxPosition1_1921680101.csv")
    # sourceData = pd.read_csv("datasets_lidar/boxPosition2/boxPosition2_1921680100.csv")
    # targetData = pd.read_csv("datasets_lidar/boxPosition2/boxPosition2_1921680101.csv")
    # sourceData = pd.read_csv("datasets_lidar/crane/crane_1921680100.csv")
    # targetData = pd.read_csv("datasets_lidar/crane/crane_1921680101.csv")

    ## remove outliers which are further away then 10 meters
    dist_sourceData = calc_distance(sourceData)
    dist_targetData = calc_distance(targetData)

    dist_threshold = 10  # 閾値

    sourceData = z_cut(sourceData, dist_sourceData, dist_threshold, -0.35)
    targetData = z_cut(targetData, dist_targetData, dist_threshold, -0.15)

    # source data
    print("Transforming source data")
    source = source_preprocess(
        sourceData,
        param.trans_init_100,
        param.x_min_100,
        param.x_max_100,
        param.y_max,
        param.outlier,
    )

    # target data
    print("Transforming target data")
    target = source_preprocess(
        targetData,
        param.trans_init_101,
        param.x_min_101,
        param.x_max_101,
        param.y_max,
        param.outlier,
    )

    # draw_registration_result(source, target, np.identity(4))  # 回転後

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
