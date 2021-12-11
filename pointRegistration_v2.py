from numpy.lib.utils import source
import pandas as pd
import open3d as o3
import numpy as np
import copy


def calc_distance(inputFrame):
    return (inputFrame["X"] ** 2 + inputFrame["Y"] ** 2 + inputFrame["Z"] ** 2) ** 0.5


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
        # sourceData = pd.read_csv("datasets_lidar/crane/crane_192168010%s.csv" % str(i))

        # remove outliers which are further away then 10 meters
        dist_sourceData = calc_distance(sourceData)  # 原点からの距離を計算

        # print(sourceData(dist_sourceData>20))
        print(np.nonzero((dist_sourceData < dist_threshold).values)[0])
        sourceData = sourceData.iloc[
            np.nonzero((dist_sourceData < dist_threshold).values)[0], :
        ]
        # print(sourceData)

        # source data
        print("Transforming source data")
        source = o3.geometry.PointCloud()
        sourceMatrix = np.array(
            [sourceData["X"], sourceData["Y"], sourceData["Z"]]
        ).transpose()
        source.points = o3.utility.Vector3dVector(sourceMatrix)

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
            ##自作
            # source_down, source_fpfh = preprocess_point_cloud(
            #     pcds[source_id], voxel_size
            # )
            # target_down, target_fpfh = preprocess_point_cloud(
            #     pcds[target_id], voxel_size
            # )
            # result_ransac = execute_global_registration(
            #     source_down, target_down, source_fpfh, target_fpfh, voxel_size
            # )
            # so.append(source_down)
            # ta.append(target_down)
            # res_ran.append(result_ransac)
            ##ここまで

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

    print("\npcd_downsは\n")
    print(pcds_down[0])
    o3.visualization.draw_geometries(
        pcds_down,
        zoom=0.5412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[-1.6172, 4.0475, -1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )

    # # 5
    # print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    # with o3.utility.VerbosityContextManager(o3.utility.VerbosityLevel.Debug) as cm:
    #     pose_graph = full_registration(
    #         pcds_down,
    #         max_correspondence_distance_coarse,
    #         max_correspondence_distance_fine,
    #     )
    # print(pose_graph)
    # # # 6
    # print("Optimizing PoseGraph ...")
    # option = o3.pipelines.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=max_correspondence_distance_fine,
    #     edge_prune_threshold=0.25,
    #     reference_node=0,
    # )
    # with o3.utility.VerbosityContextManager(o3.utility.VerbosityLevel.Debug) as cm:
    #     o3.pipelines.registration.global_optimization(
    #         pose_graph,
    #         o3.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    #         o3.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    #         option,
    #     )

    # # # 7
    # print("Transform points and display")
    # for point_id in range(len(pcds_down)):
    #     print(pose_graph.nodes[point_id].pose)
    #     pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    # # o3.visualization.draw_geometries(
    #     pcds_down,
    #     zoom=0.3412,
    #     front=[0.4257, -0.2125, -0.8795],
    #     lookat=[2.6172, 2.0475, 1.532],
    #     up=[-0.0694, -0.9768, 0.2024],
    # )

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
