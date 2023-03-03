import copy
import json
import os

import numpy as np
import open3d as o3d
import pymeshlab
from scipy.spatial.transform import Rotation


#root_path = os.path.split(os.path.split(__file__)[0])[0]
root_path = "/home/abenbihi/ws/tf/sdfstudio/outputs/shared_data/Clean_meshes/"
mesh_path = os.path.join(root_path, 'meshes')
# mesh_path = os.path.join(os.path.split(root_path)[0], 'meshes')
gt_meshes = os.path.join(os.path.join(mesh_path, 'gt_mesh'))
reconstructed_mesh = os.path.join(mesh_path, 'reconstructed_mesh')
parameter_file = os.path.join(root_path,'params')

DEBUG = (1==1)

PYMESHLAB_VERSION = "2021.10" # {2021.10, latest}

volsdf_scaled_meshes = [
        #'03_sugar_box',
        #'05_mustard_bottle',
        #'06_tuna_fish_can',
        '07_pudding_box',
        #'08_gelatin_box',
        #'10_banana',
        #'19_large_clamp',
        #'21_foam_brick'
        ]

neus_scaled_meshes = [
        #'03_sugar_box',
        #'06_tuna_fish_can',
        #'07_pudding_box',
        #'08_gelatin_box',
        #'10_banana',
        #'17_scissors',
        #'19_large_clamp',
        ]

monosdf_scaled_meshes = [
        '03_sugar_box',
        '05_mustard_bottle',
        '06_tuna_fish_can',
        '07_pudding_box',
        '10_banana',
        '17_scissors',
        '21_foam_brick'
        ]

ngp_scaled_meshes = [
        '03_sugar_box',
        '05_mustard_bottle',
        '06_tuna_fish_can',
        '10_banana',
        '17_scissors',
        '19_large_clamp'
        ]

nerfacto_scaled_meshes = [
        #'03_sugar_box',
        '05_mustard_bottle',
        #'06_tuna_fish_can',
        #'07_pudding_box',
        #'08_gelatin_box',
        #'09_large_clamp',
        #'10_banana',
        #'17_scissors',
        #'21_foam_brick'
        ]


def deg2rad(a_deg):
    return np.pi * a_deg / 180.0


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def main(method):
    print("\nMETHOD: %s"%method)

    with open("./params/%s_icp.json"%method, "r") as f:
        icp_params = json.load(f)

    method_mesh_path = os.path.join(reconstructed_mesh, method)
    original_mesh_path = os.path.join(method_mesh_path, 'original')
    print("original_mesh_path: %s"%original_mesh_path)

    # output directories
    scaled_mesh_path = os.path.join(method_mesh_path, 'scaled_mesh')
    aligned_mesh_path = os.path.join(method_mesh_path, 'aligned_mesh')
    if not os.path.exists(scaled_mesh_path):
        os.mkdir(scaled_mesh_path)
    if not os.path.exists(aligned_mesh_path):
        os.mkdir(aligned_mesh_path)

    # gather the list of ground-truth meshes
    gt_file_list = []
    for file_name in os.listdir(gt_meshes):
        if file_name[-4:] == '.ply':
            gt_file_list.append(file_name)
    
    # scaling 'to real world scale
    if method == "NGP":
        rec_meshes = [l.split(".")[0] for l in sorted(os.listdir(original_mesh_path))
                if l.split(".")[1] == "obj"]
    else:
        rec_meshes = sorted(os.listdir(original_mesh_path))
    print("\nrec_meshes:")
    print(rec_meshes)
    for file in rec_meshes:
        # load the raw reconstructed mesh
        print("processing file : ", file)
        ms = pymeshlab.MeshSet()
        scale = int(300 * 0.77)
        
        if method == "NGP":
            file_path = os.path.join(original_mesh_path, "%s.obj"%file)
        else:
            file_path = os.path.join(original_mesh_path, file, 'mesh.obj')
        print("Load reconstructed mesh from %s"%file_path)
        ms.load_new_mesh(file_path)
        
        # process it
        #https://pymeshlab.readthedocs.io/en/2021.10/filter_list.html
        if PYMESHLAB_VERSION == "2021.10":
            ms.repair_non_manifold_edges()
            ms.close_holes()
            ms.matrix_set_from_translation_rotation_scale(scalex=scale, scaley=scale, scalez=scale)
        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html
        elif PYMESHLAB_VERSION == "latest":
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes()
            ms.compute_matrix_from_translation_rotation_scale(scalex=scale, scaley=scale, scalez=scale)
        else:
            raise ValueError("Unknown PYMESHLAB_VERSION")


        # save scaled mesh
        scale_mesh_file_dir = os.path.join(scaled_mesh_path, file)
        if not os.path.exists(scale_mesh_file_dir):
            os.mkdir(scale_mesh_file_dir)
        ms.save_current_mesh(os.path.join(scale_mesh_file_dir, 'mesh.ply'), save_face_color=True, save_textures=True)

    # Aligning to ground truth frame
    scaled_meshes = sorted(os.listdir(scaled_mesh_path))
    print(scaled_meshes)
    if DEBUG:

        #scaled_meshes = neus_scaled_meshes
        #scaled_meshes = monosdf_scaled_meshes
        scaled_meshes = ngp_scaled_meshes
        scaled_meshes = nerfacto_scaled_meshes


    for file in scaled_meshes:
        print("processing file : ", file)
        # load scaled mesh
        file_path = os.path.join(scaled_mesh_path, file, 'mesh.ply')
        mesh = o3d.io.read_triangle_mesh(file_path)
        
        print("load scaled mesh: %s"%file_path)

        # prepare output filepath
        aligned_mesh = os.path.join(aligned_mesh_path, file)

        # load gt mesh
        mesh_id = int(file.split("_")[0])
        gt_file_path = "%s/obj_%06d.ply"%(gt_meshes, mesh_id)
        print("gt_file_path: ", gt_file_path)
        gt_mesh = o3d.io.read_triangle_mesh(gt_file_path)
    
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()

        #if DEBUG:
        #    with open("./params.json", "r") as f:
        #        data = json.load(f)
        #    print(data)
        #else:
        #    key = file.split(".")[0]
        #    assert(key in list(icp_params.keys()))
        #    data = icp_params[key]

        key = file.split(".")[0]
        assert(key in list(icp_params.keys()))
        data = icp_params[key]

        threshold = data["threshold"]
        criteria.max_iteration = data["criteria.max_iteration"]
        criteria.relative_rmse = data["criteria.relative_rmse"]
        criteria.relative_fitness = data["criteria.relative_fitness"]
        ax = data["ax"]
        ay = data["ay"]
        az = data["az"]
        tx = data["tx"]
        ty = data["ty"]
        tz = data["tz"]
        num_points = data["num_points"]
        ax, ay, az = [deg2rad(l) for l in [ax,ay,az]]

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)]])

        Ry = np.array([
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)]])

        Rz = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1]])

        trans_init = np.eye(4)
        trans_init[:3,:3] = Rx @ Ry @ Rz
        trans_init[:3,3] = np.array([tx, ty, tz])

        # sub-sampled the scaled reconstructed mesh and the groundtruth one
        pcd_mesh = mesh.sample_points_uniformly(number_of_points=num_points)
        pcd_gt_mesh = gt_mesh.sample_points_uniformly(number_of_points=num_points)

        draw_registration_result(pcd_mesh, pcd_gt_mesh, trans_init)

        print("\nApply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_mesh, pcd_gt_mesh, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria
        )
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(pcd_mesh, pcd_gt_mesh, reg_p2p.transformation)


        # transform the scaled mesh
        T = reg_p2p.transformation
        translation_matrix = T[:3, 3:]
        rotation_matrix = T[:3, :3]
        rotation_matrix = np.linalg.inv(rotation_matrix)
        r = Rotation.from_matrix(rotation_matrix)
        angles = r.as_euler("xyz", degrees=True)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)

        if PYMESHLAB_VERSION == "2021.10":
            ms.repair_non_manifold_edges()
            ms.close_holes()
            ms.matrix_set_from_translation_rotation_scale(translationx=translation_matrix[0],
                                                              translationy=translation_matrix[1],
                                                              translationz=translation_matrix[2],
                                                              rotationx=angles[0],
                                                              rotationy=angles[1],
                                                              rotationz=angles[2]
                                                              )
        elif PYMESHLAB_VERSION == "latest":
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes()
            ms.compute_matrix_from_translation_rotation_scale(translationx=translation_matrix[0],
                                                              translationy=translation_matrix[1],
                                                              translationz=translation_matrix[2],
                                                              rotationx=angles[0],
                                                              rotationy=angles[1],
                                                              rotationz=angles[2]
                                                              )
        else:
            raise ValueError("Unknown PYMESHLAB_VERSION")

        
        # save aligned mesh
        align_mesh_file_dir = os.path.join(aligned_mesh_path, file)
        if not os.path.exists(align_mesh_file_dir):
            os.mkdir(align_mesh_file_dir)
        ms.save_current_mesh(os.path.join(align_mesh_file_dir, 'mesh.ply'), save_face_color=True, save_textures=True)



def scale_mesh(method):
    print("\nMETHOD: %s"%method)

    method_mesh_path = os.path.join(reconstructed_mesh, method)
    original_mesh_path = os.path.join(method_mesh_path, 'original')
    print("original_mesh directorty: %s"%original_mesh_path)

    # output directories
    scaled_mesh_path = os.path.join(method_mesh_path, 'scaled_mesh')
    if not os.path.exists(scaled_mesh_path):
        os.mkdir(scaled_mesh_path)
    print("Saving meshes to %s"%scaled_mesh_path)

    # collect list of reconstructed meshes
    if method == "NGP":
        rec_meshes = [l.split(".")[0] for l in sorted(os.listdir(original_mesh_path))
                if l.split(".")[1] == "obj"]
    else:
        rec_meshes = sorted(os.listdir(original_mesh_path))
    print("\nrec_meshes:")
    print(rec_meshes)
    for file in rec_meshes:
        # load the raw reconstructed mesh
        #print("processing file : ", file)
        ms = pymeshlab.MeshSet()
        scale = int(300 * 0.77)
        
        if method == "NGP":
            file_path = os.path.join(original_mesh_path, "%s.obj"%file)
        else:
            file_path = os.path.join(original_mesh_path, file, 'mesh.obj')
        print("\nLoad reconstructed mesh from %s"%file_path)
        ms.load_new_mesh(file_path)
        
        # process it
        #https://pymeshlab.readthedocs.io/en/2021.10/filter_list.html
        if PYMESHLAB_VERSION == "2021.10":
            ms.repair_non_manifold_edges()
            ms.close_holes()
            ms.matrix_set_from_translation_rotation_scale(scalex=scale, scaley=scale, scalez=scale)
        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html
        elif PYMESHLAB_VERSION == "latest":
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes()
            ms.compute_matrix_from_translation_rotation_scale(scalex=scale, scaley=scale, scalez=scale)
        else:
            raise ValueError("Unknown PYMESHLAB_VERSION")

        # save scaled mesh
        scale_mesh_file_dir = os.path.join(scaled_mesh_path, file)
        if not os.path.exists(scale_mesh_file_dir):
            os.mkdir(scale_mesh_file_dir)
        output_path = os.path.join(scale_mesh_file_dir, 'mesh.ply')
        ms.save_current_mesh(output_path, save_face_color=True, save_textures=True)
        print("Saving scaled mesh to %s"%output_path)


def align_mesh(method):
    print("\nMETHOD: %s"%method)

    with open("./params/%s_icp.json"%method, "r") as f:
        icp_params = json.load(f)

    method_mesh_path = os.path.join(reconstructed_mesh, method)
    original_mesh_path = os.path.join(method_mesh_path, 'original')
    scaled_mesh_path = os.path.join(method_mesh_path, 'scaled_mesh')
    print("original_mesh_path: %s"%original_mesh_path)

    # output directories
    aligned_mesh_path = os.path.join(method_mesh_path, 'aligned_mesh')
    if not os.path.exists(aligned_mesh_path):
        os.mkdir(aligned_mesh_path)
    print("Saving meshes to %s"%aligned_mesh_path)

    ## gather the list of ground-truth meshes
    #gt_file_list = []
    #for file_name in os.listdir(gt_meshes):
    #    if file_name[-4:] == '.ply':
    #        gt_file_list.append(file_name)
    
    # Aligning to ground truth frame
    scaled_meshes = sorted(os.listdir(scaled_mesh_path))
    print("\nscaled_meshes to process:")
    print(scaled_meshes)

    if DEBUG:
        #scaled_meshes = neus_scaled_meshes
        #scaled_meshes = monosdf_scaled_meshes
        #scaled_meshes = ngp_scaled_meshes
        #scaled_meshes = nerfacto_scaled_meshes
        scaled_meshes = volsdf_scaled_meshes

    for file in scaled_meshes:
        #print("processing file : ", file)
        # load scaled mesh
        file_path = os.path.join(scaled_mesh_path, file, 'mesh.ply')
        mesh = o3d.io.read_triangle_mesh(file_path)
        print("load scaled mesh: %s"%file_path)

        # prepare output filepath
        aligned_mesh = os.path.join(aligned_mesh_path, file)

        # load gt mesh
        mesh_id = int(file.split("_")[0])
        gt_file_path = "%s/obj_%06d.ply"%(gt_meshes, mesh_id)
        print("gt_file_path: ", gt_file_path)
        gt_mesh = o3d.io.read_triangle_mesh(gt_file_path)
    
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()

        key = file.split(".")[0]
        assert(key in list(icp_params.keys()))
        data = icp_params[key]

        threshold = data["threshold"]
        criteria.max_iteration = data["criteria.max_iteration"]
        criteria.relative_rmse = data["criteria.relative_rmse"]
        criteria.relative_fitness = data["criteria.relative_fitness"]
        ax = data["ax"]
        ay = data["ay"]
        az = data["az"]
        tx = data["tx"]
        ty = data["ty"]
        tz = data["tz"]
        num_points = data["num_points"]
        ax, ay, az = [deg2rad(l) for l in [ax,ay,az]]

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)]])

        Ry = np.array([
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)]])

        Rz = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1]])

        trans_init = np.eye(4)
        trans_init[:3,:3] = Rx @ Ry @ Rz
        trans_init[:3,3] = np.array([tx, ty, tz])

        # sub-sampled the scaled reconstructed mesh and the groundtruth one
        pcd_mesh = mesh.sample_points_uniformly(number_of_points=num_points)
        pcd_gt_mesh = gt_mesh.sample_points_uniformly(number_of_points=num_points)

        draw_registration_result(pcd_mesh, pcd_gt_mesh, trans_init)

        print("\nApply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_mesh, pcd_gt_mesh, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria
        )
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(pcd_mesh, pcd_gt_mesh, reg_p2p.transformation)

        # transform the scaled mesh
        T = reg_p2p.transformation
        translation_matrix = T[:3, 3:]
        rotation_matrix = T[:3, :3]
        rotation_matrix = np.linalg.inv(rotation_matrix)
        r = Rotation.from_matrix(rotation_matrix)
        angles = r.as_euler("xyz", degrees=True)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)

        if PYMESHLAB_VERSION == "2021.10":
            ms.repair_non_manifold_edges()
            ms.close_holes()
            ms.matrix_set_from_translation_rotation_scale(translationx=translation_matrix[0],
                                                              translationy=translation_matrix[1],
                                                              translationz=translation_matrix[2],
                                                              rotationx=angles[0],
                                                              rotationy=angles[1],
                                                              rotationz=angles[2]
                                                              )
        elif PYMESHLAB_VERSION == "latest":
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes()
            ms.compute_matrix_from_translation_rotation_scale(translationx=translation_matrix[0],
                                                              translationy=translation_matrix[1],
                                                              translationz=translation_matrix[2],
                                                              rotationx=angles[0],
                                                              rotationy=angles[1],
                                                              rotationz=angles[2]
                                                              )
        else:
            raise ValueError("Unknown PYMESHLAB_VERSION")

        
        # save aligned mesh
        align_mesh_file_dir = os.path.join(aligned_mesh_path, file)
        if not os.path.exists(align_mesh_file_dir):
            os.mkdir(align_mesh_file_dir)
        output_path = os.path.join(align_mesh_file_dir, 'mesh.ply')
        ms.save_current_mesh(output_path, save_face_color=True, save_textures=True)

        print("Saving aligned mesh to %s"%output_path)



if __name__=="__main__":

    method = ['neus', 'volsdf', 'monosdf', 'NGP', 'nerfacto']

    #main(method[2])
    
    #scale_mesh(method[1])
    align_mesh(method[1])
