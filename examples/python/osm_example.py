import copy
import time

import clipperpy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def generate_dataset(pcd, pcd2, n1, n2, nia, noa, sigma, T_21):
    """Generate Dataset"""

    # pcd = o3d.io.read_point_cloud(pcfile)

    # if nia > n1:
    #     raise ValueError(
    #         "Cannot have more inlier associations "
    #         "than there are model points. Increase"
    #         "the number of points to sample from the"
    #         "original point cloud model."
    #     )

    # radius of outlier sphere
    R = 200

    # Downsample from the original point cloud, sample randomly
    I = np.random.choice(len(pcd.points), n1, replace=False)
    D1 = np.asarray(pcd.points).T

    # Rotate into view 2 using ground truth transformation
    I = np.random.choice(len(pcd2.points), n2, replace=False)
    D2 = np.asarray(pcd2.points).T
    D2 = T_21[0:3, 0:3] @ D2 + T_21[0:3, 3].reshape(-1, 1)

    # Add noise uniformly sampled from a sigma cube around the true point
    eta1 = np.random.uniform(low=-sigma / 2.0, high=sigma / 2.0, size=D1.shape)
    eta2 = np.random.uniform(low=-sigma / 2.0, high=sigma / 2.0, size=D2.shape)

    # Add noise to view 2
    D1 += eta1
    D2 += eta2

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
    Abad = np.zeros((n1 * n2 - n1, 2))
    itr = 0
    for i in range(n1):
        for j in range(n2):
            if i == j:
                continue
            Abad[itr, :] = [i, j]
            itr += 1

    # Sample good and bad associations to satisfy total
    # num of associations with the requested outlier ratio
    IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
    IAbad = np.random.choice(Abad.shape[0], noa, replace=False)
    A = np.concatenate((Agood[IAgood, :], Abad[IAbad, :])).astype(np.int32)

    # Ground truth associations
    Agt = Agood[IAgood, :]

    return (D1, D2, Agt, A)


# def generate_Abad(n1, n2):
#     print("STARTING ABAD")
#     # Create a grid of all index combinations
#     I, J = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

#     # Flatten the arrays to get all combinations as column vectors
#     I_flat = I.flatten()
#     J_flat = J.flatten()

#     # Filter out the pairs where indices are the same
#     mask = I_flat != J_flat
#     Abad = np.vstack((I_flat[mask], J_flat[mask])).T

#     print("Finished ABAD")
#     return Abad


# def generate_dataset(pcd_1, pcd_2, n1, n2, nia, noa, T_21):
#     """Generate Dataset"""
#     if nia > n1:
#         raise ValueError(
#             "Cannot have more inlier associations "
#             "than there are model points. Increase"
#             "the number of points to sample from the"
#             "original point cloud model."
#         )

#     # Get points from pcds
#     D1 = np.asarray(pcd_1.points).T
#     D2 = np.asarray(pcd_2.points).T

#     # Rotate into view 2 using ground truth transformation
#     D2_tf = T_21[0:3, 0:3] @ D2 + T_21[0:3, 3].reshape(-1, 1)

#     # Correct associations to draw from
#     Agood = np.tile(np.arange(nia).reshape(-1, 1), (1, 2))

#     # Incorrect association to draw from
#     Abad = generate_Abad(n1, n2)
#     # print("STARTING ABAD")
#     # Abad = np.zeros((n1 * n2 - nia, 2))
#     # itr = 0
#     # for i in range(n1):
#     #     for j in range(n2):
#     #         if i == j:
#     #             continue
#     #         Abad[itr, :] = [i, j]
#     #         itr += 1
#     # print("Finished ABAD")

#     # Sample good and bad associations to satisfy total
#     # num of associations with the requested outlier ratio
#     IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
#     IAbad = np.random.choice(Abad.shape[0], noa, replace=False)
#     A = np.concatenate((Agood[IAgood, :], Abad[IAbad, :])).astype(np.int32)
#     # Ground truth associations
#     Agt = Agood[IAgood, :]

#     return (D1, D2_tf, Agt, A)


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


def read_npy_file(file_path):
    point_cloud = np.load(file_path, allow_pickle=True)
    return point_cloud


def get_transformed_point_cloud(
    pc2, transformation_matrices, frame_number1, frame_number2
):
    """Transform a point cloud from origin to where the lidar sensor is, given a scan number.

    Args:
        pc (np array of points): The lidar pointcloud to be transformed.
        transformation_matrices (dict of poses): Python dictionary containing the 3x3 pose of lidar sensor for a given scan number.
        frame_number (int): The nth lidar scan (scan number) within the specific sequence.

    Returns:
        transformed_xyz: The pointcloud transformed to where the lidar sensor is.
    """
    # Get the transformation matrix for the current frame
    transformation_matrix_1 = transformation_matrices.get(frame_number1)
    transformation_matrix_2 = transformation_matrices.get(frame_number2)

    # Calculate the inverse of the first matrix
    inverse_matrix_1 = np.linalg.inv(transformation_matrix_1)

    # Calculate the transformation difference
    transformation_difference = np.dot(inverse_matrix_1, transformation_matrix_2)

    # Separate the XYZ coordinates and intensity values
    xyz = pc2[:, :3]

    # Convert the XYZ coordinates to homogeneous coordinates
    xyz_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

    # Apply the transformation to each XYZ coordinate
    transformed_xyz = np.dot(xyz_homogeneous, transformation_difference.T)[:, :3]

    return transformed_xyz


def read_poses(file_path):
    """
    Reads 4x4 or 3x4 transformation matrices from a file.

    Args:
        file_path (str): The path to the file containing poses.

    Returns:
        dict: A dictionary where keys are frame indices and values are 4x4 numpy arrays representing transformation matrices.
    """
    poses_xyz = {}
    with open(file_path, "r") as file:
        for line in file:
            elements = line.strip().split()
            frame_index = int(elements[0])
            if len(elements[1:]) == 16:
                matrix_4x4 = np.array(elements[1:], dtype=float).reshape((4, 4))
            else:
                matrix_3x4 = np.array(elements[1:], dtype=float).reshape((3, 4))
                matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])
            poses_xyz[frame_index] = matrix_4x4
    return poses_xyz


def labels2RGB2(label_ids, labels_dict):
    # Prepare the output array
    rgb_array = np.zeros((label_ids.shape[0], 3), dtype=float)
    for idx, label_id in enumerate(label_ids):
        # print(f"label_id: {label_id}")
        if label_id in labels_dict:
            color = labels_dict.get(label_id, (0, 0, 0))  # Default color is black
            # print(f"Color: {color}")
            rgb_array[idx] = np.array(color) / 255
            # print(f"rgb_array[idx]: {rgb_array[idx]}")
    return rgb_array


import os

from kitti360_OSM_utils import labels

labels_dict = {label.id: label.color for label in labels}

pointcloud1_frame_number = 259
pointcloud2_frame_number = 269
pcd_1_build_file = f"/home/donceykong/Desktop/datasets/KITTI-360/data_3d_extracted/2013_05_28_drive_0000_sync/map_segments/0000000009_0000000500_build_points_map.npy"
pcd_1_road_file = f"/home/donceykong/Desktop/datasets/KITTI-360/data_3d_extracted/2013_05_28_drive_0000_sync/map_segments/0000000009_0000000500_road_points_map.npy"
pcd_2_build_file = f"/home/donceykong/Desktop/datasets/KITTI-360/data_3d_extracted/2013_05_28_drive_0000_sync/map_segments/0000000509_0000001000_build_points_map.npy"
pcd_2_road_file = f"/home/donceykong/Desktop/datasets/KITTI-360/data_3d_extracted/2013_05_28_drive_0000_sync/map_segments/0000000509_0000001000_road_points_map.npy"

seq = 0
sequence_dir = f"2013_05_28_drive_{seq:04d}_sync"
velodyne_poses_file = os.path.join(
    "/home/donceykong/Desktop/datasets/KITTI-360/data_poses",
    sequence_dir,
    "velodyne_poses.txt",
)
velodyne_poses = read_poses(velodyne_poses_file)

pointcloud_1_build = read_npy_file(pcd_1_build_file)[:, :3]
pointcloud_1_road = read_npy_file(pcd_1_road_file)[:, :3]
pointcloud_1_combined = np.concatenate((pointcloud_1_build, pointcloud_1_road), axis=0)
# rgb_np = labels2RGB2(lidar_points_3d_accum[:, 4], seq_extract.labels_dict)
pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(pointcloud_1_combined)
pcd_1.paint_uniform_color(np.array([1, 0, 0]))

pointcloud_2_build = read_npy_file(pcd_2_build_file)[:, :3]
pointcloud_2_road = read_npy_file(pcd_2_road_file)[:, :3]
pointcloud_2_combined = np.concatenate((pointcloud_2_build, pointcloud_2_road), axis=0)
pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(pointcloud_2_combined)
pcd_2.paint_uniform_color(np.array([0, 1, 0]))

o3d.visualization.draw_geometries(
    [
        pcd_2,
        pcd_1,
    ]
)

# Generate dataset
n1 = 1000  # number of points used on model (i.e., seen in view 1)
n2o = 2000  # number of outliers in data (i.e., seen in view 2)
n2 = n1 + n2o  # number of points in view 2
sigma = 0.01  # uniform noise [m] range

m = 2000  # total number of associations in problem
outrat = 0.9  # outlier ratio of initial association set
noa = round(m * outrat)  # number of outlier associations
nia = m - noa  # number of inlier associations

# generate random (R,t)
T_21 = np.eye(4)
T_21[0:3, 0:3] = Rotation.random().as_matrix()
T_21[0:3, 3] = np.random.uniform(low=-5, high=5, size=(3,))

D1, D2, Agt, A = generate_dataset(pcd_2, pcd_1, n1, n2, nia, noa, sigma, T_21)

# View unaligned data
pcd_2_tf = o3d.geometry.PointCloud()
pcd_2_tf.points = o3d.utility.Vector3dVector(D2.T)
pcd_2_tf.paint_uniform_color(np.array([0, 1, 0]))
o3d.visualization.draw_geometries([pcd_1, pcd_2_tf])

#
#
iparams = clipperpy.invariants.EuclideanDistanceParams()
iparams.sigma = 0.01
iparams.epsilon = 0.1
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

# A = clipper.get_initial_associations()
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
model.paint_uniform_color(np.array([0, 1, 0]))


data = o3d.geometry.PointCloud()
data.points = o3d.utility.Vector3dVector(D2.T)
data.paint_uniform_color(np.array([1, 0, 0]))

# corr = o3d.geometry.LineSet.create_from_point_cloud_correspondences(model, data, Ain)
# o3d.visualization.draw_geometries([model, data, corr])

p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
That_21 = p2p.compute_transformation(model, data, o3d.utility.Vector2iVector(Ain))
get_err(T_21, That_21)

draw_registration_result(model, data, That_21)
