import statistics
from numpy.lib.utils import source
import pandas as pd
import open3d as o3
import numpy as np
import copy

import chair_parameter as param

# 複数点
# http://www.open3d.org/docs/0.13.0/tutorial/pipelines/multiway_registration.html
# Global registartion
# http://www.open3d.org/docs/0.13.0/tutorial/pipelines/global_registration.html


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
        sourceMatrix[0] = sourceMatrix[0] + 2
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
    sourceMatrix[0] = sourceMatrix[0] - medX
    sourceMatrix[1] = sourceMatrix[1] - medY
    sourceMatrix[2] = sourceMatrix[2] - medZ

    sourceMatrix = sourceMatrix.T

    source.points = o3.utility.Vector3dVector(sourceMatrix)

    return source


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return pcd_down


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    print("Loading files")

    pcds = []
    dist_threshold = 10  # 距離の閾値(原点から遠すぎるものを排除？)
    for i in range(4):
        # read the data
        sourceData = pd.read_csv("datasets_lidar/chair/chair_192168010%s.csv" % str(i))
        # sourceData = pd.read_csv(
        #     "datasets_lidar/boxPosition1/boxPosition1_192168010%s.csv" % str(i)
        # )
        # sourceData = pd.read_csv(
        #     "datasets_lidar/boxPosition2/boxPosition2_192168010%s.csv" % str(i)
        # )

        # remove outliers which are further away then 10 meters
        dist_sourceData = calc_distance(sourceData)  # 原点からの距離を計算

        # print(sourceData(dist_sourceData>20))
        sourceData = z_cut(sourceData, dist_sourceData, dist_threshold, param.z_min_101)
        # print(sourceData)

        # source data
        print("Transforming source data")
        source = source_preprocess(
            sourceData,
            param.transarray[i],
            param.x_min,
            param.x_max,
            param.y_min,
            param.y_max,
            param.outlier,
        )

        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = source.voxel_down_sample(voxel_size)

        ##この4行入れたらicp_reguration_errorが出なくなった？
        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        pcds.append(pcd_down)
    # print(pcds)
    return pcds


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        np.identity(4),
        o3.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    icp_fine = o3.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    transformation_icp = icp_fine.transformation
    information_icp = (
        o3.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine, icp_fine.transformation
        )
    )
    return transformation_icp, information_icp


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


def full_registration(
    pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine
):
    pose_graph = o3.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    # so = []
    # ta = []
    # res_ran = []
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id]
            )
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    # draw_registration_result(so, ta, res_ran.transformation)
    return pose_graph


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


if __name__ == "__main__":
    voxel_size = 0.05  # means 5cm for the dataset
    pcds_down = prepare_dataset(voxel_size)

    # 入力の点群を準備したやつ表示
    # print("\npcd_downsは\n")
    print(pcds_down)
    o3.visualization.draw_geometries(
        pcds_down,
        zoom=0.6559,
        front=[-0.5452, -0.836, -0.2011],
        lookat=[0, 0, 0],
        up=[-0.2779, -0.282, 0.1556],
    )

    # # 5
    # print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3.utility.VerbosityContextManager(o3.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(
            pcds_down,
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine,
        )
    print(pose_graph)

    # # 6
    print("Optimizing PoseGraph ...")
    option = o3.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    with o3.utility.VerbosityContextManager(o3.utility.VerbosityLevel.Debug) as cm:
        o3.pipelines.registration.global_optimization(
            pose_graph,
            o3.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

    # # 7
    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3.visualization.draw_geometries(
        pcds_down,
        zoom=0.6559,
        front=[-0.5452, -0.836, -0.2011],
        lookat=[0, 0, 0],
        up=[-0.2779, -0.282, 0.1556],
    )

    # result_ransac = execute_global_registration(
    #     source_down, target_down, source_fpfh, target_fpfh, voxel_size
    # )
    # print(result_ransac)

    # draw_registration_result(source_down, target_down,
    #                          result_ransac.transformation)

    # this does not work yet - error
    # result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
    #                                  voxel_size)
    # print(result_icp)
    # draw_registration_result(source, target, result_icp.transformation)

    ###今はここは使ってない


# def execute_global_registration(
#     source_down, target_down, source_fpfh, target_fpfh, voxel_size
# ):
#     distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
#     result = o3.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down,
#         target_down,
#         source_fpfh,
#         target_fpfh,
#         True,
#         distance_threshold,
#         o3.pipelines.registration.TransformationEstimationPointToPoint(False),
#         3,
#         [
#             o3.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#             o3.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold
#             ),
#         ],
#         o3.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
#     )
#     return result


# def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 0.4
#     print(":: Point-to-plane ICP registration is applied on original point")
#     print("   clouds to refine the alignment. This time we use a strict")
#     print("   distance threshold %.3f." % distance_threshold)
#     result = o3.pipelines.registration.registration_icp(
#         source,
#         target,
#         distance_threshold,
#         result_ransac.transformation,
#         o3.pipelines.registration.TransformationEstimationPointToPlane(),
#     )
#     return result


###
