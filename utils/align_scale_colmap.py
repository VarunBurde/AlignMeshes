import open3d as o3d
import copy
import numpy as np
import os
import pymeshlab
import json


root_path = os.path.split(os.path.split(__file__)[0])[0]
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


if __name__=="__main__":
    main()
