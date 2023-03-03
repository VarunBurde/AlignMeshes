import open3d as o3d
import copy
import numpy as np
import os
import pymeshlab
import json


root_path = os.path.split(os.path.split(__file__)[0])[0]
#root_path = "/home/abenbihi/ws/tf/sdfstudio/outputs/shared_data/Clean_meshes/"
mesh_path = os.path.join(root_path, 'meshes')
gt_meshes = os.path.join(os.path.join(mesh_path, 'gt_mesh'))
reconstructed_mesh = os.path.join(mesh_path, 'reconstructed_mesh')
colmap_mesh_path = os.path.join(reconstructed_mesh, 'colmap')
original_mesh_path = os.path.join(colmap_mesh_path, 'original')
scaled_mesh_path = os.path.join(colmap_mesh_path, 'scaled_mesh')
aligned_mesh_path = os.path.join(colmap_mesh_path, 'aligned_mesh')
parameter_file = os.path.join(root_path,'params')
colmap_param_file = os.path.join(parameter_file, 'colmap_icp.json')

DEBUG = (0==1)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def filter_noise(bb_pcd):
    mesh_out = bb_pcd.filter_smooth_simple(number_of_iterations=5)
    return mesh_out

def remove_floating_part(mesh):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
    mesh_0.remove_triangles_by_mask(triangles_to_remove)

    return mesh_0


def deg2rad(a_deg):
    return np.pi * a_deg / 180.0


def main():
    with open(colmap_param_file, "r") as f:
        icp_params = json.load(f)

    #print("original_mesh_path: ", original_mesh_path)
    print("\ngt_meshes: ", gt_meshes)
    gt_file_list = []
    for file_name in sorted(os.listdir(gt_meshes)):
        if file_name[-4:] == '.ply':
            gt_file_list.append(file_name)

    #print("gt_file_list")
    #print(gt_file_list)

    # scaling 'to real world scale
    rec_meshes = sorted(os.listdir(original_mesh_path))
    print("\nrec_meshes")
    print(rec_meshes)

    if (1==1):
        for file in rec_meshes:
            print("processing file : ", file)
            file_obj = file.replace('.ply', '.obj')
            #scale = int(300 * 0.77) # TODO: not used?

            # path to raw reconstructed mesh
            file_path = os.path.join(original_mesh_path, file)

            # load raw reconstructured mesh
            mesh = o3d.io.read_triangle_mesh(file_path)
            box = o3d.geometry.OrientedBoundingBox()
            crop_box = box.get_axis_aligned_bounding_box()
            crop_box.min_bound = [-300, -300, -300.0]
            crop_box.max_bound = [300, 300, -4.5]
            # crop the reconstructed mesh
            bb_pcd = mesh.crop(crop_box)
            bb_pcd.compute_vertex_normals()

            # write the cropped reconstructured mesh
            output_path = os.path.join(scaled_mesh_path, file_obj)
            print("Writing cropped mesh to %s"%output_path)
            o3d.io.write_triangle_mesh(mesh=bb_pcd,
                    filename=output_path, write_triangle_uvs=True)


    # Aligning to ground truth frame
    scaled_meshes = sorted(os.listdir(scaled_mesh_path))
    print("\nscaled_meshes:")
    print(scaled_meshes)

    if DEBUG:
        scaled_meshes = [
                #'03_sugar_box.obj',
                #'05_mustard_bottole.obj',
                #'06_tuna_fish_can.obj',
                #'07_pudding_box.obj',
                #'08_gelatin_box.obj',
                #'10_banana.obj',
                #'17_scissors.obj',
                #'19_large_clamp.obj',
                '21_foam_brick.obj'
                ]

    for file in scaled_meshes:
        print("processing file : ", file)

        # load scaled mesh
        file_path = os.path.join(scaled_mesh_path, file)
        mesh = o3d.io.read_triangle_mesh(file_path)

        # prepare output filepath
        aligned_mesh = os.path.join(aligned_mesh_path, file)

        # load gt mesh
        mesh_id = int(file.split("_")[0])
        gt_file_path = "%s/obj_%06d.ply"%(gt_meshes, mesh_id)
        print("gt_file_path: ", gt_file_path)
        gt_mesh = o3d.io.read_triangle_mesh(gt_file_path)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria()

        if DEBUG:
            with open("./params.json", "r") as f:
                data = json.load(f)
            print(data)
        else:
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


def scale_mesh(method):
    print("\nMETHOD: %s"%method)

    method_mesh_path = os.path.join(reconstructed_mesh, method)
    original_mesh_path = os.path.join(method_mesh_path, 'original')
    scaled_mesh_path = os.path.join(method_mesh_path, 'scaled_mesh')

    # scaling 'to real world scale
    if method == "NGP":
        rec_meshes = [l.split(".")[0] for l in sorted(os.listdir(original_mesh_path))
                if l.split(".")[1] == "obj"]
    else:
        rec_meshes = sorted(os.listdir(original_mesh_path))
    print("\nrec_meshes")
    print(rec_meshes)

    for file in rec_meshes:
        print("processing file : ", file)
        file_obj = file.replace('.ply', '.obj')
        #scale = int(300 * 0.77) # TODO: not used?

        # path to raw reconstructed mesh
        #file_path = os.path.join(original_mesh_path, file)
        if method == "NGP":
            file_path = os.path.join(original_mesh_path, file)
        else:
            file_path = os.path.join(original_mesh_path, file)
        print("Load reconstructed mesh from %s"%file_path)

        # load raw reconstructured mesh
        mesh = o3d.io.read_triangle_mesh(file_path)
        box = o3d.geometry.OrientedBoundingBox()
        crop_box = box.get_axis_aligned_bounding_box()

        if method =="colmap":
            crop_box.min_bound = [-300, -300, -300.0]
            crop_box.max_bound = [300, 300, -4.5]


        if method == "NGP":
            multiple = 1.7
            crop_box.min_bound = [-1.0 * multiple, -1.0 * multiple, -1.0 * multiple]
            crop_box.max_bound = [1.0 * multiple, -0.08 * multiple, 1.0 * multiple]

        # crop the reconstructed mesh
        bb_pcd = mesh.crop(crop_box)
        bb_pcd.compute_vertex_normals()

        # filter noise with laplacian
        bb_pcd = filter_noise(bb_pcd)

        # filter small floating part in space
        bb_pcd = remove_floating_part(bb_pcd)

        o3d.visualization.draw_geometries([bb_pcd])

        # write the cropped reconstructured mesh
        scale_mesh_file_dir = os.path.join(scaled_mesh_path, file)
        if not os.path.exists(scale_mesh_file_dir):
            os.makedirs(scale_mesh_file_dir)
        output_path = os.path.join(scale_mesh_file_dir, 'mesh.obj')
        #ms.save_current_mesh(output_path, save_face_color=True, save_textures=True)
        #print("Saving scaled mesh to %s"%output_path)
        if method == "NGP":
            bb_pcd.scale(300/4, center=(0, 0, 0))


        #output_path = os.path.join(scaled_mesh_path, file_obj)
        print("Writing cropped mesh to %s"%output_path)
        o3d.io.write_triangle_mesh(mesh=bb_pcd,
                filename=output_path, write_triangle_uvs=True)


        
def align_mesh(method):
    print("\nMETHOD: %s"%method)
    parameter = os.path.join(parameter_file,"%s_icp.json"%method )
    with open(parameter, "r") as f:
        icp_params = json.load(f)

    method_mesh_path = os.path.join(reconstructed_mesh, method)
    original_mesh_path = os.path.join(method_mesh_path, 'original')
    scaled_mesh_path = os.path.join(method_mesh_path, 'scaled_mesh')
    print("original_mesh directorty: %s"%original_mesh_path)

    # output directories
    aligned_mesh_path = os.path.join(method_mesh_path, 'aligned_mesh')
    if not os.path.exists(aligned_mesh_path):
        os.mkdir(aligned_mesh_path)
    print("Saving meshes to %s"%aligned_mesh_path)

    # Aligning to ground truth frame
    scaled_meshes = sorted(os.listdir(scaled_mesh_path))
    print("\nscaled_meshes to process:")
    print(scaled_meshes)

    if DEBUG:
        scaled_meshes = [
                #'03_sugar_box.obj',
                #'05_mustard_bottole.obj',
                #'06_tuna_fish_can.obj',
                #'07_pudding_box.obj',
                #'08_gelatin_box.obj',
                #'10_banana.obj',
                #'17_scissors.obj',
                #'19_large_clamp.obj',
                '21_foam_brick.obj'
                ]
        
        ngp_scaled_meshes = [
                '03_sugar_box',
                #'05_mustard_bottle',
                #'06_tuna_fish_can',
                #'10_banana',
                #'17_scissors',
                #'19_large_clamp'
                ]

        scaled_meshes = ngp_scaled_meshes

    for file in scaled_meshes:
        print("processing file : ", file)

        # load scaled mesh
        file_path = os.path.join(scaled_mesh_path, file)
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

        if DEBUG:
            with open("./params.json", "r") as f:
                data = json.load(f)
            print(data)
        else:
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
        mesh.transform(reg_p2p.transformation)

        o3d.io.write_triangle_mesh(mesh=mesh, filename=aligned_mesh, write_triangle_uvs=True)
        print("Saving aligned mesh to %s"%aligned_mesh)

if __name__=="__main__":
    method = ['colmap', 'NGP']

    #main(method[2])
    
    # scale_mesh(method[0])
    align_mesh(method[0])
