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


def distance_cut(sourceData, dist_data, dist_threshold):
    distance_cut = sourceData.iloc[
        np.nonzero((dist_data < dist_threshold).values)[0], :
    ]
    # distance_cut = distance_cut.iloc[np.nonzero((distance_cut["Z"] < 0.8).values)[0], :]
    # preprocessed_data = distance_cut.iloc[
    #     np.nonzero((distance_cut["Z"] > z_threshold).values)[0], :
    # ]

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
    outlier,
):
    source = o3.geometry.PointCloud()  # generate point_cloud
    sourceMatrix = np.array([sourceData["X"], sourceData["Y"], sourceData["Z"]])

    # rotate
    sourceMatrix = np.dot(trans_init_z, sourceMatrix)
    sourceMatrix = np.dot(trans_init_x, sourceMatrix)
    sourceMatrix = np.dot(trans_init_y, sourceMatrix)

    medX = statistics.median(sourceMatrix[0])
    medY = statistics.median(sourceMatrix[1])
    medZ = statistics.median(sourceMatrix[2])
    print("中央値は\n")
    print(medX)
    print(medY)
    print(medZ)
    print("\n")
    # Transfer
    if medX < 0:
        sourceMatrix[0] = sourceMatrix[0] + 2
    if medY < 0:
        sourceMatrix[1] = sourceMatrix[1] + 3.5
    # if medZ < 0.1:
    #     sourceMatrix[2] = sourceMatrix[2] + 0.2

    # Cut
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
    # medX = statistics.median(sourceMatrix[0])
    # medY = statistics.median(sourceMatrix[1])
    # medZ = statistics.median(sourceMatrix[2])
    # print(medX)
    # print(medY)
    # print(medZ)
    # sourceMatrix[0] = sourceMatrix[0] - medX
    # sourceMatrix[1] = sourceMatrix[1] - medY
    # sourceMatrix[2] = sourceMatrix[2] - medZ

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
        zoom=0.5559,
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


def prepare_dataset(voxel_size, i, m, n):
    print(":: Load two point clouds and disturb initial pose.")
    print("Loading files")
    # 0→boxBinBrikets,1→box1,2→box2,3→chair,4→crane,5→rubbishBin,
    # 6→rubbishBin_bricks,7→two_Bricks
    sourceData = pd.read_csv(param.readData_multi[i] % str(m))
    targetData = pd.read_csv(param.readData_multi[i] % str(n))

    ## remove outliers which are further away then 10 meters
    dist_sourceData = calc_distance(sourceData)
    dist_targetData = calc_distance(targetData)

    dist_threshold = 10  # 閾値

    # cut by (height) and distance from origin
    sourceData = distance_cut(sourceData, dist_sourceData, dist_threshold)
    targetData = distance_cut(targetData, dist_targetData, dist_threshold)

    # source data
    print("Transforming source data")
    # rotation and remove unnecessary data
    source = source_preprocess(
        sourceData,
        param.transarray_z[m],
        param.transarray_x[m],
        param.transarray_y[m],
        param.x_min,
        param.x_max,
        param.y_min,
        param.y_max,
        param.z_max,
        param.z_min[m],
        param.outlier,
    )

    # target data
    print("Transforming target data")
    # rotation and remove unnecessary data
    target = source_preprocess(
        targetData,
        param.transarray_z[n],
        param.transarray_x[n],
        param.transarray_y[n],
        param.x_min,
        param.x_max,
        param.y_min,
        param.y_max,
        param.z_max,
        param.z_min[n],
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
    voxel_size = 0.04  # means 5cm for the dataset
    i = 3
    m = 0
    n = 3
    (
        source,
        target,
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
    ) = prepare_dataset(voxel_size, i, m, n)

    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    print("Global_registration後")
    print(result_ransac)
    draw_registration_result(source, target, result_ransac.transformation)

    # print(source)
    # print(target)
    # print(source_down)
    # print(target_down)
    # print(source_fpfh)
    # print(target_fpfh)

    # # this does not work yet - error(refine)
    refine_icp = refine_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    print("refine後")
    print(refine_icp)
    # draw_registration_result(source, target, refine_icp.transformation)

    # # rapid_global
    # start = time.time()
    # result_fast = execute_fast_global_registration(
    #     source_down, target_down, source_fpfh, target_fpfh, voxel_size
    # )
    # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    # print(result_fast)
    # draw_registration_result(source_down, target_down, result_fast.transformation)

    threshold = 0.02
    trans_init = np.eye(4)

    # # Point to point ICP
    # icp(source, target, threshold, result_ransac.transformation)
    # icp(source, target, threshold, refine_icp.transformation)

    # ICP回数多め
    # icp_more(source, target, threshold, trans_init)

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
    #     source_down,
    #     target_down,
    #     threshold,
    #     trans_init,
    #     o3.pipelines.registration.TransformationEstimationForColoredICP(),
    # )
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # draw_registration_result(source, target, reg_p2l.transformation)
