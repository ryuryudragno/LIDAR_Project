import time
import pandas as pd
import open3d as o3
import numpy as np
import copy
import statistics

import chair_parameter as param
import probreg

# import euler
# command qで画面消去→VScodeに焦点が当たってると全部消えるので注意
# 椅子を左向きになるように合わせる(101(or100)に合わせてる？)
# shift + option + 9で · が打てる


def calc_distance(inputFrame):
    return (inputFrame["X"] ** 2 + inputFrame["Y"] ** 2 + inputFrame["Z"] ** 2) ** 0.5


def z_cut(sourceData, dist_data, dist_threshold, z_threshold):
    distance_cut = sourceData.iloc[
        np.nonzero((dist_data < dist_threshold).values)[0], :
    ]
    distance_cut = distance_cut.iloc[np.nonzero((distance_cut["Z"] < 0.8).values)[0], :]
    preprocessed_data = distance_cut.iloc[
        np.nonzero((distance_cut["Z"] > z_threshold).values)[0], :
    ]

    return preprocessed_data


# rotation and remove unnecessary data
def source_preprocess(sourceData, trans_init, x_min, x_max, y_min, y_max, outlier):
    source = o3.geometry.PointCloud()  # generate point_cloud
    sourceMatrix = np.array([sourceData["X"], sourceData["Y"], sourceData["Z"]])

    # ここから回転 and cut
    sourceMatrix = np.dot(trans_init, sourceMatrix)

    medX = statistics.median(sourceMatrix[0])
    medY = statistics.median(sourceMatrix[1])

    if medX < 0:
        sourceMatrix[0] = sourceMatrix[0] + 2.03
    if medY < 0:
        sourceMatrix[1] = sourceMatrix[1] + 3.5

    sourceMatrix = np.where(
        (sourceMatrix[0] > x_min) & (sourceMatrix[0] < x_max), sourceMatrix, outlier
    )
    sourceMatrix = np.where(
        (sourceMatrix[1] > y_min) & (sourceMatrix[1] < y_max), sourceMatrix, outlier
    )
    # axisで行か列かを指定できる
    sourceMatrix = sourceMatrix[:, np.all(sourceMatrix != outlier, axis=0)]
    medX = statistics.median(sourceMatrix[0])
    medY = statistics.median(sourceMatrix[1])
    medZ = statistics.median(sourceMatrix[2])
    print(medX)
    print(medY)
    print(medZ)
    # if medY > 1.8:
    #     sourceMatrix[2] = sourceMatrix[2] + 0
    # if medZ > 0.2:
    #     sourceMatrix[2] = sourceMatrix[2] - 0.2

    sourceMatrix[0] = sourceMatrix[0] - medX
    sourceMatrix[1] = sourceMatrix[1] - medY
    sourceMatrix[2] = sourceMatrix[2] - medZ

    sourceMatrix = sourceMatrix.T

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
        zoom=0.6559,
        front=[-0.5452, -0.836, -0.2011],
        lookat=[0, 0, 0],
        up=[-0.2779, -0.282, 0.1556],
    )


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 4
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

    sourceData = pd.read_csv("datasets_lidar/boxPosition1/boxPosition1_1921680100.csv")
    targetData = pd.read_csv("datasets_lidar/boxPosition1/boxPosition1_1921680101.csv")
    # sourceData = pd.read_csv("datasets_lidar/boxPosition2/boxPosition2_1921680100.csv")
    # targetData = pd.read_csv("datasets_lidar/boxPosition2/boxPosition2_1921680101.csv")
    # sourceData = pd.read_csv("datasets_lidar/crane/crane_1921680100.csv")
    # targetData = pd.read_csv("datasets_lidar/crane/crane_1921680101.csv")

    ## remove outliers which are further away then 10 meters
    dist_sourceData = calc_distance(sourceData)
    dist_targetData = calc_distance(targetData)

    dist_threshold = 10  # 閾値

    # cut by height and distance from origin
    sourceData = z_cut(sourceData, dist_sourceData, dist_threshold, param.z_min_100)
    targetData = z_cut(targetData, dist_targetData, dist_threshold, param.z_min_101)

    # source data
    print("Transforming source data")
    # rotation and remove unnecessary data
    source = source_preprocess(
        sourceData,
        param.trans_init_100,
        param.x_min,
        param.x_max,
        param.y_min,
        param.y_max,
        param.outlier,
    )

    # target data
    print("Transforming target data")
    # rotation and remove unnecessary data
    target = source_preprocess(
        targetData,
        param.trans_init_101,
        param.x_min,
        param.x_max,
        param.y_min,
        param.y_max,
        param.outlier,
    )
    print(target)
    # 事前処理後
    draw_registration_result(source, target, np.identity(4))

    # 点群をダウンサンプリングし特徴を抽出
    # downsample the point cloud, estimate normals, then compute a FPFH feature for each point)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # print("sourceは" + str(source))
    # print("source_downは" + str(source_down))
    # print("source_fpfhは" + str(source_fpfh))
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


def icp(source, target, threshold, trans_init):
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


def icp_more(source, target, threshold, trans_init):
    reg_p2p = o3.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3.pipelines.registration.TransformationEstimationPointToPoint(),
        o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000),
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)


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
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    # print(source)
    # print(target)
    # print(source_down)
    # print(target_down)
    # print(source_fpfh)
    # print(target_fpfh)

    # # this does not work yet - error(refine)
    # result_icp = refine_registration(
    #     source_down, target_down, source_fpfh, target_fpfh, voxel_size
    # )
    # print(result_icp)
    # draw_registration_result(source, target, result_icp.transformation)

    # 高速グローバル
    # start = time.time()
    # result_fast = execute_fast_global_registration(
    #     source_down, target_down, source_fpfh, target_fpfh, voxel_size
    # )
    # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    # print(result_fast)
    # draw_registration_result(source_down, target_down, result_fast.transformation)

    threshold = 0.02
    trans_init = np.eye(4)

    # Point to point ICP
    icp(source, target, threshold, trans_init)

    # ICP回数多め
    icp_more(source, target, threshold, trans_init)

    # probreg
    # tf_param, _, _ = probreg.cpd.registration_cpd(source, target)
    # result = copy.deepcopy(source)
    # result.points = tf_param.transform(result.points)

    # # draw result
    # source.paint_uniform_color([1, 0, 0])
    # target.paint_uniform_color([0, 1, 0])
    # result.paint_uniform_color([0, 0, 1])
    # o3.visualization.draw_geometries([source, target, result])


# # Point to Plane ICP
# print("Apply point-to-plane ICP")
# reg_p2l = o3.pipelines.registration.registration_icp(
#     source,
#     target,
#     threshold,
#     trans_init,
#     o3.pipelines.registration.TransformationEstimationForColoredICP(),
# )
# print(reg_p2l)
# print("Transformation is:")
# print(reg_p2l.transformation)
# draw_registration_result(source, target, reg_p2l.transformation)
