import open3d as o3d
import copy
import numpy as np
import os
import pymeshlab
from scipy.spatial.transform import Rotation

method = ['neus', 'volsdf', 'monosdf']

root_path = os.getcwd()
mesh_path = os.path.join(os.path.split(root_path)[0], 'meshes')
gt_meshes = os.path.join(os.path.join(mesh_path, 'gt_mesh'))
reconstructed_mesh = os.path.join(mesh_path, 'reconstructed_mesh')
method_mesh_path = os.path.join(reconstructed_mesh, 'volsdf')
original_mesh_path = os.path.join(method_mesh_path, 'original')
scaled_mesh_path = os.path.join(method_mesh_path, 'scaled_mesh')
aligned_mesh_path = os.path.join(method_mesh_path, 'aligned_mesh')

if not os.path.exists(scaled_mesh_path):
    os.mkdir(scaled_mesh_path)

if not os.path.exists(aligned_mesh_path):
    os.mkdir(aligned_mesh_path)


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
    ms = pymeshlab.MeshSet()
    scale = int(300 * 0.77)
    file_path = os.path.join(original_mesh_path, file, 'mesh.obj')
    ms.load_new_mesh(file_path)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_close_holes()
    ms.compute_matrix_from_translation_rotation_scale(scalex=scale, scaley=scale, scalez=scale)
    scale_mesh_file_dir = os.path.join(scaled_mesh_path, file)
    if not os.path.exists(scale_mesh_file_dir):
        os.mkdir(scale_mesh_file_dir)
    ms.save_current_mesh(os.path.join(scale_mesh_file_dir, 'mesh.ply'), save_face_color=True, save_textures=True)

# Aligning to ground truth frame
for file in os.listdir(scaled_mesh_path):
    print("processing file : ", file)
    file_path = os.path.join(scaled_mesh_path, file, 'mesh.ply')
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
    # draw_registration_result(pcd_mesh, pcd_gt_mesh, trans_init)

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
    # draw_registration_result(pcd_mesh, pcd_gt_mesh, reg_p2p.transformation)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_close_holes()

    T = reg_p2p.transformation

    translation_matrix = T[:3, 3:]
    rotation_matrix = T[:3, :3]
    rotation_matrix = np.linalg.inv(rotation_matrix)
    r = Rotation.from_matrix(rotation_matrix)
    angles = r.as_euler("xyz", degrees=True)
    ms.compute_matrix_from_translation_rotation_scale(translationx=translation_matrix[0],
                                                      translationy=translation_matrix[1],
                                                      translationz=translation_matrix[2],
                                                      rotationx=angles[0],
                                                      rotationy=angles[1],
                                                      rotationz=angles[2]
                                                      )
    align_mesh_file_dir = os.path.join(aligned_mesh_path, file)
    if not os.path.exists(align_mesh_file_dir):
        os.mkdir(align_mesh_file_dir)
    ms.save_current_mesh(os.path.join(align_mesh_file_dir, 'mesh.ply'), save_face_color=True, save_textures=True)
