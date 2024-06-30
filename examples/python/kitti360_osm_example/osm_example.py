import copy
import os
import re
import time

import clipperpy
import numpy as np
import open3d as o3d
from kitti360_osm_utils import labels
from scipy.spatial.transform import Rotation


def generate_Abad(n1, n2):
    print("STARTING ABAD")
    # Create a grid of all index combinations
    I, J = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

    # Flatten the arrays to get all combinations as column vectors
    I_flat = I.flatten()
    J_flat = J.flatten()

    # Filter out the pairs where indices are the same
    mask = I_flat != J_flat
    Abad = np.vstack((I_flat[mask], J_flat[mask])).T

    print("Finished ABAD")
    return Abad


def generate_dataset(pcd, pcd2, n1, n2, nia, noa, sigma, T_21):
    """Generate Dataset"""

    print(f"Size of n1 is {n1} and size of n2 is {n2}.")
    if nia > n1:
        raise ValueError(
            "Cannot have more inlier associations "
            "than there are model points. Increase"
            "the number of points to sample from the"
            "original point cloud model."
        )

    # radius of outlier sphere
    R = 200

    # Downsample from the original point cloud, sample randomly
    # I1 = np.random.choice(len(pcd.points), n1, replace=False)
    # D1 = np.asarray(pcd.points)[I1, :].T
    D1 = np.asarray(pcd.points).T

    # Rotate into view 2 using ground truth transformation
    # I2 = np.random.choice(len(pcd2.points), n2, replace=False)
    # D2 = np.asarray(pcd2.points)[I2, :].T
    D2 = np.asarray(pcd2.points).T
    D2 = T_21[0:3, 0:3] @ D2 + T_21[0:3, 3].reshape(-1, 1)

    # Add noise uniformly sampled from a sigma cube around the true point
    eta1 = np.random.uniform(low=-sigma / 2.0, high=sigma / 2.0, size=D1.shape)
    eta2 = np.random.uniform(low=-sigma / 2.0, high=sigma / 2.0, size=D2.shape)

    # Add noise to view 2
    # D1 += eta1
    # D2 += eta2

    def randsphere(m, n, r):
        from scipy.special import gammainc

        X = np.random.randn(m, n)
        s2 = np.sum(X**2, axis=1)
        X = X * np.tile(
            (r * (gammainc(n / 2, s2 / 2) ** (1 / n)) / np.sqrt(s2)).reshape(-1, 1),
            (1, n),
        )
        return X

    # Add outliers to view 2
    O2 = randsphere(n2o, 3, R).T + D2.mean(axis=1).reshape(-1, 1)
    # D2 = np.hstack((D2, O2))

    # Correct associations to draw from
    Agood = np.tile(np.arange(n1).reshape(-1, 1), (1, 2))

    # Incorrect association to draw from
    # Abad = np.zeros((n1 * n2 - n1, 2))
    # itr = 0
    # for i in range(n1):
    #     for j in range(n2):
    #         if i == j:
    #             continue
    #         Abad[itr, :] = [i, j]
    #         itr += 1

    Abad = generate_Abad(n1, n2)
    # Sample good and bad associations to satisfy total
    # num of associations with the requested outlier ratio
    IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
    IAbad = np.random.choice(Abad.shape[0], noa, replace=False)
    A = np.concatenate((Agood[IAgood, :], Abad[IAbad, :])).astype(np.int32)

    # Ground truth associations
    Agt = Agood[IAgood, :]

    return (D1, D2, Agt, A)


def get_err(T, That):
    Terr = np.linalg.inv(T) @ That
    rerr = abs(np.arccos(min(max(((Terr[0:3, 0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3, 3])
    return (rerr, terr)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def labels2RGB2(label_ids, labels_dict):
    # Prepare the output array
    rgb_array = np.zeros((label_ids.shape[0], 3), dtype=np.double)
    for idx, label_id in enumerate(label_ids):
        if label_id in labels_dict:
            color = labels_dict.get(label_id, (0, 0, 0))  # Default color is black
            rgb_array[idx] = np.array(color) / 255.0
    return rgb_array


# Iterate over all files in the directory
def get_frame_numbers(directory_path):
    frame_numbers = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".npy") and "road" in filename:
            # Extract the frame number using regex
            match = re.search(r"frame_(\d+)_road_points.npy", filename)
            if match:
                frame_numbers.append(int(match.group(1)))
    return sorted(frame_numbers)


def get_accum_points(first_frame_idx, final_frame_idx, data_dir, osm_frames):
    semantic_classes = ["build", "road"]
    build_points_list = []
    road_points_list = []
    for frame_idx in range(first_frame_idx, final_frame_idx):
        for semantic_class in semantic_classes:
            frame_number = osm_frames[frame_idx]
            points_filename = f"frame_{frame_number:010d}_{semantic_class}_points.npy"
            points_filepath = os.path.join(data_dir, points_filename)
            points = np.load(points_filepath, allow_pickle=True)
            if "build" in semantic_class:
                build_points_list.extend(points)
            elif "road" in semantic_class:
                road_points_list.extend(points)
    build_accum_points = np.asarray(build_points_list)
    road_accum_points = np.asarray(road_points_list)
    return build_accum_points, road_accum_points


labels_dict = {label.id: label.color for label in labels}

seq = 0
data_dir = f"../../data/kitti360_osm_data/2013_05_28_drive_{seq:04d}_sync/"

osm_frames = get_frame_numbers(data_dir)
print(f"len(osm_frames): {len(osm_frames)}")

target_pointcloud_building, target_pointcloud_road = get_accum_points(
    first_frame_idx=0,
    final_frame_idx=len(osm_frames) - 1,
    data_dir=data_dir,
    osm_frames=osm_frames,
)
target_pointcloud_building_xyz = target_pointcloud_building[:, :3]
target_pointcloud_road_xyz = target_pointcloud_road[:, :3]
target_pointcloud_building_semantic = target_pointcloud_building[:, 4]
target_pointcloud_road_semantic = target_pointcloud_road[:, 4]
target_pointcloud_xyz_combined = np.concatenate(
    (target_pointcloud_building_xyz, target_pointcloud_road_xyz), axis=0
)
target_pointcloud_semantic_combined = np.concatenate(
    (target_pointcloud_building_semantic, target_pointcloud_road_semantic), axis=0
)
target_pointcloud_rgb = labels2RGB2(target_pointcloud_semantic_combined, labels_dict)
target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(target_pointcloud_xyz_combined)
target_pcd.colors = o3d.utility.Vector3dVector(target_pointcloud_rgb)

source_pointcloud_building, source_pointcloud_road = get_accum_points(
    first_frame_idx=0,
    final_frame_idx=150,
    data_dir=data_dir,
    osm_frames=osm_frames,
)
source_pointcloud_building_xyz = source_pointcloud_building[:, :3]
source_pointcloud_road_xyz = source_pointcloud_road[:, :3]
source_pointcloud_combined = np.concatenate(
    (source_pointcloud_building_xyz, source_pointcloud_road_xyz), axis=0
)
source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(source_pointcloud_combined)
source_pcd.paint_uniform_color(np.array([0, 1, 0]))

o3d.visualization.draw_geometries([target_pcd, source_pcd])

# Generate dataset
percent_input_points = 0.01
# number of points used on model (This is the number of points in target?)
n1 = round(percent_input_points * len(target_pcd.points))
# number of outliers in data (points from both target and source that do not overlap)
n2o = round(percent_input_points * len(source_pcd.points))
n2 = n1 + n2o  # number of points in view 2
sigma = 0.01  # uniform noise [m] range

m = n1  # total number of associations in problem
outrat = 0.5  # outlier ratio of initial association set
noa = round(m * outrat)  # number of outlier associations
nia = m - noa  # number of inlier associations

# generate random (R,t)
T_21 = np.eye(4)
T_21[0:3, 0:3] = Rotation.random().as_matrix()
T_21[0:3, 3] = np.random.uniform(low=-5, high=5, size=(3,))

D1, D2, Agt, A = generate_dataset(target_pcd, source_pcd, n1, n2, nia, noa, sigma, T_21)

# View unaligned data
pcd_2_tf = o3d.geometry.PointCloud()
pcd_2_tf.points = o3d.utility.Vector3dVector(D2.T)
pcd_2_tf.paint_uniform_color(np.array([0, 1, 0]))
o3d.visualization.draw_geometries([target_pcd, pcd_2_tf])

#
iparams = clipperpy.invariants.EuclideanDistanceParams()
iparams.sigma = 0.01
iparams.epsilon = 0.01
invariant = clipperpy.invariants.EuclideanDistance(iparams)

params = clipperpy.Params()
params.rounding = clipperpy.Rounding.DSD_HEU
clipper = clipperpy.CLIPPER(invariant, params)

t0 = time.perf_counter()
clipper.score_pairwise_consistency(D1, D2, A)
t1 = time.perf_counter()
print(f"Affinity matrix creation took {t1-t0:.3f} seconds")

t0 = time.perf_counter()
clipper.solve()
t1 = time.perf_counter()

A = clipper.get_initial_associations()
Ain = clipper.get_selected_associations()

p = np.isin(Ain, Agt)[:, 0].sum() / Ain.shape[0]
r = np.isin(Ain, Agt)[:, 0].sum() / Agt.shape[0]
print(
    f"CLIPPER selected {Ain.shape[0]} inliers from {A.shape[0]} "
    f"putative associations (precision {p:.2f}, recall {r:.2f}) in {t1-t0:.3f} s"
)

# 1st point cloud (road)
model = o3d.geometry.PointCloud()
model.points = o3d.utility.Vector3dVector(D1.T)
model.colors = o3d.utility.Vector3dVector(target_pointcloud_rgb)


data = o3d.geometry.PointCloud()
data.points = o3d.utility.Vector3dVector(D2.T)
data.paint_uniform_color(np.array([0, 1, 0]))

# Draw pc correspondances
corr = o3d.geometry.LineSet.create_from_point_cloud_correspondences(model, data, Ain)
o3d.visualization.draw_geometries([model, data, corr])

p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
That_21 = p2p.compute_transformation(model, data, o3d.utility.Vector2iVector(Ain))
get_err(T_21, That_21)

draw_registration_result(model, data, That_21)
