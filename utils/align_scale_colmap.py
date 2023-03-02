import open3d as o3d
import copy
import numpy as np
import os
import pymeshlab

root_path = os.getcwd()
mesh_path = os.path.join(os.path.split(root_path)[0], 'meshes')
gt_meshes = os.path.join(os.path.join(mesh_path, 'gt_mesh'))
reconstructed_mesh = os.path.join(mesh_path, 'reconstructed_mesh')
colmap_mesh_path = os.path.join(reconstructed_mesh, 'colmap')
original_mesh_path = os.path.join(colmap_mesh_path, 'original')
scaled_mesh_path = os.path.join(colmap_mesh_path, 'scaled_mesh')
aligned_mesh_path = os.path.join(colmap_mesh_path, 'aligned_mesh')


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


gt_file_list = []
for file_name in os.listdir(gt_meshes):
    if file_name[-4:] == '.ply':
        gt_file_list.append(file_name)

# scaling 'to real world scale
for file in os.listdir(original_mesh_path):
    print("processing file : ", file)
    file_obj = file.replace('.ply', '.obj')
    scale = int(300 * 0.77)
    file_path = os.path.join(original_mesh_path, file)
    mesh = o3d.io.read_triangle_mesh(file_path)
    box = o3d.geometry.OrientedBoundingBox()
    crop_box = box.get_axis_aligned_bounding_box()
    crop_box.min_bound = [-300, -300, -300.0]
    crop_box.max_bound = [300, 300, -4.5]
    bb_pcd = mesh.crop(crop_box)
    bb_pcd.compute_vertex_normals()
    o3d.io.write_triangle_mesh(mesh=bb_pcd, filename=os.path.join(scaled_mesh_path, file_obj), write_triangle_uvs=True)

# Aligning to ground truth frame
for file in os.listdir(scaled_mesh_path):
    print("processing file : ", file)
    file_path = os.path.join(scaled_mesh_path, file)
    mesh = o3d.io.read_triangle_mesh(file_path)
    aligned_mesh = os.path.join(aligned_mesh_path, file)
    # find gt mesh
    for gt_name in gt_file_list:
        if gt_name[-6:-4] == file[0:2]:
            gt_file_path = os.path.join(gt_meshes, gt_name)
            gt_mesh = o3d.io.read_triangle_mesh(gt_file_path)

    pcd_mesh = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_gt_mesh = gt_mesh.sample_points_uniformly(number_of_points=5000)

    trans_init = np.eye(4)
    draw_registration_result(pcd_mesh, pcd_gt_mesh, trans_init)

    threshold = 10
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
    criteria.max_iteration = 3000
    criteria.relative_rmse = 0.00000001
    criteria.relative_fitness = 0.00000001

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_mesh, pcd_gt_mesh, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(pcd_mesh, pcd_gt_mesh, reg_p2p.transformation)
    mesh.transform(reg_p2p.transformation)

    o3d.io.write_triangle_mesh(mesh=mesh, filename=aligned_mesh, write_triangle_uvs=True)
