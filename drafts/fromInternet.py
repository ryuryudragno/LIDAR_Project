import open3d as o3d
import numpy as np
import copy
import pandas as pd
import pymeshlab

# csvをplyの形と同じにできてるのか不明
# https://qiita.com/Sumire_sn24/items/6d6c182d055015e4fcf7


def calc_distance(inputFrame):
    return (inputFrame["X"] ** 2 + inputFrame["Y"] ** 2 + inputFrame["Z"] ** 2) ** 0.5


def prepare_data(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    print("Loading files")

    pcds = []
    dist_threshold = 10  # 距離の閾値(原点から遠すぎるものを排除？)
    for i in range(2):
        # read the data
        sourceData = pd.read_csv(
            "datasets_lidar/boxPosition1/boxPosition1_192168010%s.csv" % str(i)
        )

        dist_sourceData = calc_distance(sourceData)  # 原点からの距離を計算

        # print(sourceData(dist_sourceData>20))
        print(np.nonzero((dist_sourceData < dist_threshold).values)[0])
        sourceData = sourceData.iloc[
            np.nonzero((dist_sourceData < dist_threshold).values)[0], :
        ]

        # source data
        print("Transforming source data")
        source = o3d.geometry.PointCloud()
        sourceMatrix = np.array(
            [sourceData["X"], sourceData["Y"], sourceData["Z"]]
        ).transpose()
        source.points = o3d.utility.Vector3dVector(sourceMatrix)

        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = source.voxel_down_sample(voxel_size)

        ##この4行入れたらicp_reguration_errorが出なくなった？
        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        pcds.append(pcd_down)
    return pcds


# 法線の計算と色付け
def add_color_normal(pcd):
    pcd.paint_uniform_color(np.random.rand(3))
    # pcd.estimate_normals(kdt_n)


def load_pcds(pcd_files):
    print(":: Load pcd files, size = %d" % len(pcd_files))
    pcds = []
    for file in pcd_files:
        pcd = o3d.io.read_point_cloud(file)
        add_color_normal(pcd)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
    return pcds


def merge(pcds):
    all_points = []
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))

    return merged_pcd


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556],
    )


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500),
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
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def refine_registration(
    source, target, result_ransac, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh


def register(source, target, voxel_size):
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        source, target, voxel_size
    )

    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    # result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    result = refine_registration(
        source, target, result_ransac, source_fpfh, target_fpfh, voxel_size
    )

    source.transform(result.transformation)
    # draw_registration_result(source, target)
    merge_pcd = merge([source, target])
    return merge_pcd


def create_mesh_by_meshlab(mesh_filename):
    ms = pymeshlab.MeshSet()
    result_filename = "result_by_meshlab.ply"
    ms.load_new_mesh(mesh_filename)
    print(":: Create mesh by MeshLab")
    ms.apply_filter("compute_normals_for_point_sets")
    ms.apply_filter("surface_reconstruction_screened_poisson")
    ms.apply_filter("remove_isolated_pieces_wrt_diameter", mincomponentdiag=50)
    ms.save_current_mesh(result_filename)

    return result_filename


if __name__ == "__main__":
    # input_files = [
    #     "datasets_lidar/boxPosition1/boxPosition1_1921680101.csv",
    #     "datasets_lidar/boxPosition1/boxPosition1_1921680102.csv",
    # ]
    voxel_size = 0.5  # 初期は0.05
    pcds = prepare_data(voxel_size)
    # pcds = load_pcds(input_files)

    n_pcds = len(pcds)
    source_pcd = pcds[0]
    # voxel_size = np.abs((source_pcd.get_max_bound() - source_pcd.get_min_bound())).max() / 30

    for pcd_id in range(1, n_pcds):
        source_pcd = register(source_pcd, pcds[pcd_id], voxel_size)
        print(":: Registration result, times %d" % pcd_id)
        source_pcd.estimate_normals()
        source_pcd.paint_uniform_color(np.random.rand(3))
        o3d.visualization.draw_geometries([source_pcd])

    # move to origin
    source_pcd = source_pcd.translate(-source_pcd.get_center())

    # create meth by meshlab
    result_pcd_filename = "result_pcd.ply"
    o3d.io.write_point_cloud(result_pcd_filename, source_pcd)
    result_mesh_filename = create_mesh_by_meshlab(result_pcd_filename)

    result_mesh = o3d.io.read_triangle_mesh(result_mesh_filename, print_progress=True)
    result_mesh.paint_uniform_color(np.random.rand(3))
    result_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([result_mesh])
