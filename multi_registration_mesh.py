import statistics
from matplotlib import pyplot as plt
from numpy.lib.utils import source
import pandas as pd
import open3d as o3
import numpy as np
import copy

import chair_parameter as param
import read_video_csv as vd

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
    # medX = statistics.median(sourceMatrix[0])
    # medY = statistics.median(sourceMatrix[1])
    # medZ = statistics.median(sourceMatrix[2])
    # sourceMatrix[0] = sourceMatrix[0] - medX
    # sourceMatrix[1] = sourceMatrix[1] - medY
    # sourceMatrix[2] = sourceMatrix[2] - medZ

    sourceMatrix = sourceMatrix.T

    source.points = o3.utility.Vector3dVector(sourceMatrix)
    # source.transform(trans_carib)

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


def prepare_dataset(voxel_size, num):
    print(":: Load two point clouds and disturb initial pose.")
    print("Loading files")

    sources = []
    pcds = []
    dist_threshold = 10  # 距離の閾値(原点から遠すぎるものを排除？)
    for i in range(4):
        # read the data
        # 0→boxBinBrikets,1→box1,2→box2,3→chair,4→crane,5→rubbishBin,
        # 6→rubbishBin_bricks,7→two_Bricks
        sourceData = pd.read_csv(param.readData_multi[num] % str(i))
        # sourceData = pd.read_csv(vd.readData_test[0] % str(i))

        # remove outliers which are further away then 10 meters
        dist_sourceData = calc_distance(sourceData)  # 原点からの距離を計算

        # print(sourceData(dist_sourceData>20))
        sourceData = distance_cut(sourceData, dist_sourceData, dist_threshold)
        # print(sourceData)

        # source data
        print("Transforming source data")
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

        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = source.voxel_down_sample(voxel_size)

        ##この4行入れたらicp_reguration_errorが出なくなった？
        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        sources.append(source)
        pcds.append(pcd_down)
    # print(pcds)
    return sources, pcds


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
                    # 逆行列を求めている
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


def Alpha(pcd_combined_down, alpha):
    mesh = o3.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd_combined_down, alpha
    )
    mesh.compute_vertex_normals()
    # o3.visualization.draw_geometries(
    #     [mesh],
    #     mesh_show_back_face=True,
    #     zoom=0.5158999999999998,
    #     front=[-0.77920625806744848, -0.57987940671965754, 0.23786021325766751],
    #     lookat=[-0.12481240763005762, 1.0039607063096871, 0.81194244371714996],
    #     up=[0.19116025605760037, 0.1415469754318206, 0.97129923826290332],
    # )

    # # look for good alpha
    # LookForAlpha(pcd_combined_down, alpha)

    return mesh


def LookForAlpha(pcd_combined_down, alpha):
    tetra_mesh, pt_map = o3.geometry.TetraMesh.create_from_point_cloud(
        pcd_combined_down
    )
    for alpha in np.logspace(np.log10(0.1), np.log10(0.015), num=4):
        print(f"alpha={alpha:.3f}")
        mesh = o3.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd_combined_down, alpha, tetra_mesh, pt_map
        )
        mesh.compute_vertex_normals()
        o3.visualization.draw_geometries(
            [mesh],
            mesh_show_back_face=True,
            zoom=0.6559,
            front=[-0.5452, -0.736, -0.3011],
            lookat=[0, 0, 0],
            up=[-0.2779, -0.282, 0.2556],
        )


def BallPivot(mesh):
    pcd = mesh.sample_points_poisson_disk(3000)

    radii = [0.02, 0.03, 0.04, 0.05]
    rec_mesh = o3.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3.utility.DoubleVector(radii)
    )
    o3.visualization.draw_geometries(
        [pcd, rec_mesh],
        zoom=0.6559,
        front=[-0.5452, -0.736, -0.3011],
        lookat=[0, 0, 0],
        up=[-0.2779, -0.282, 0.2556],
    )


def Poisson(pcd_combined_down):
    with o3.utility.VerbosityContextManager(o3.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_combined_down, depth=6
        )
    print(mesh)
    o3.visualization.draw_geometries(
        [mesh],
        zoom=0.664,
        front=[-0.5452, -0.736, -0.3011],
        lookat=[0, 0, 0],
        up=[-0.2779, -0.282, 0.2556],
    )
    return mesh, densities


def VisualizeDensity(mesh, densities):
    densities = np.asarray(densities)
    density_colors = plt.get_cmap("plasma")(
        (densities - densities.min()) / (densities.max() - densities.min())
    )
    density_colors = density_colors[:, :3]
    density_mesh = o3.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3.utility.Vector3dVector(density_colors)
    o3.visualization.draw_geometries(
        [density_mesh],
        zoom=0.664,
        front=[-0.4761, -0.4698, -0.7434],
        lookat=[1.8900, 3.2596, 0.9284],
        up=[0.2304, -0.8825, 0.4101],
    )
    return densities


def RemoveLowDensity(mesh, densities):
    vertices_to_remove = densities < np.quantile(densities, 0.7)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    o3.visualization.draw_geometries(
        [mesh],
        zoom=0.664,
        front=[-0.5452, -0.736, -0.3011],
        lookat=[0, 0, 0],
        up=[-0.2779, -0.282, 0.2556],
    )


if __name__ == "__main__":
    voxel_size = 0.02  # means 5cm for the dataset
    # 0→boxBinBrikets,1→box1,2→box2,3→chair,4→crane,5→rubbishBin,
    # 6→rubbishBin_bricks,7→two_Bricks
    n = 3
    sources, pcds_down = prepare_dataset(voxel_size, n)

    print("Before preprocessing ...")
    # o3.visualization.draw_geometries(
    #     pcds_down,
    #     zoom=0.5158999999999998,
    #     front=[-0.77920625806744848, -0.57987940671965754, 0.23786021325766751],
    #     lookat=[-0.12481240763005762, 1.0039607063096871, 0.81194244371714996],
    #     up=[0.19116025605760037, 0.1415469754318206, 0.97129923826290332],
    # )

    # # 5
    print("Full registration ...")
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
    pcd_combined = o3.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        # print(pose_graph.edges[point_id])
        pcds_down[point_id].transform(param.trans_carib[point_id])
        pcd_combined += pcds_down[point_id]

    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    print(pcd_combined_down)
    o3.visualization.draw_geometries(
        [pcd_combined_down],
        # width=1920,
        # height=720,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False,
        zoom=0.78190000000000004,
        front=[-0.50022870208461867, 0.74264254806709906, 0.44525643331808762],
        lookat=[1.1721829566284132, 1.5976718356891839, -0.31953885726192527],
        up=[0.17665673501636506, -0.41587238921020825, 0.89210008063330581],
    )

    # # Alpha shapes
    alpha = 0.025
    print(f"alpha={alpha:.3f}")
    mesh = Alpha(pcd_combined_down, alpha)

    # # Ball pivoting
    # BallPivot(mesh)

    # # Poisson surface reconstruction
    # print("run Poisson surface reconstruction")
    # mesh, densities = Poisson(pcd_combined_down)

    # print("visualize densities")
    # densities = VisualizeDensity(mesh, densities)

    # print("remove low density vertices")
    # RemoveLowDensity(mesh, densities)
