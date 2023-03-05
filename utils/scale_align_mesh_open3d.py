import copy
import json
import os
import trimesh
import numpy as np
import open3d as o3d
import pymeshlab
from scipy.spatial.transform import Rotation
import cv2


root_path = os.path.split(os.path.split(__file__)[0])[0]
mesh_path = os.path.join(root_path, 'meshes')
gt_meshes = os.path.join(os.path.join(mesh_path, 'gt_mesh_obj'))
reconstructed_mesh = os.path.join(mesh_path, 'reconstructed_mesh')
parameter_file = os.path.join(root_path,'params')


def remove_outlier_with_estimated_bbox_gt(mesh, gt_mesh, method):
    if method == "colmap":
        aabb = gt_mesh.get_oriented_bounding_box()
        aabb.scale(1.08, aabb.get_center())
        mesh = mesh.crop(aabb)
        o3d.visualization.draw_geometries([mesh])
        return mesh
    else:
        aabb = gt_mesh.get_oriented_bounding_box()
        aabb.scale(1.08, aabb.get_center())
        mesh = mesh.crop(aabb)
        o3d.visualization.draw_geometries([mesh])
        return mesh


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def deg2rad(a_deg):
    return np.pi * a_deg / 180.0

def remove_small_floating_part(mesh):
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
    o3d.visualization.draw_geometries([mesh_0])
    return mesh_0

def remove_floating_part(mesh):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_1 = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_1.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([mesh_1])
    return mesh_1


def apply_transformation(method, file, mesh, gt_mesh):
    # o3d.visualization.draw_geometries([gt_mesh])

    ## Loading icp parmaeter
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
    parameter = os.path.join(parameter_file, "%s_icp.json" % method)
    with open(parameter, "r") as f:
        icp_params = json.load(f)
    key = file.split(".")[0]
    assert (key in list(icp_params.keys()))
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
    ax, ay, az = [deg2rad(l) for l in [ax, ay, az]]

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
    trans_init[:3, :3] = Rx @ Ry @ Rz
    trans_init[:3, 3] = np.array([tx, ty, tz])


    # sub-sampled the scaled reconstructed mesh and the groundtruth one
    # pcd_mesh = mesh.sample_points_uniformly(number_of_points=4000)
    # pcd_gt_mesh = gt_mesh.sample_points_uniformly(number_of_points=4000)
    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = mesh.vertices
    pcd_mesh.colors = o3d.utility.Vector3dVector(np.array(mesh.vertex_colors) * 1.8)
    pcd_mesh.normals = mesh.vertex_normals

    pcd_gt_mesh = o3d.geometry.PointCloud()
    pcd_gt_mesh.points = gt_mesh.vertices
    pcd_gt_mesh.colors = gt_mesh.vertex_colors
    pcd_gt_mesh.normals = gt_mesh.vertex_normals

    if method in ['colmap', 'NGP']:

        voxel_radius = 0.04
        source_down = pcd_mesh.voxel_down_sample(voxel_radius)
        target_down = pcd_gt_mesh.voxel_down_sample(voxel_radius)

        draw_registration_result_original_color(source_down, target_down, trans_init)

        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius * 2, max_nn=30))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_radius * 2, max_nn=30))

        print("\nApply point-to-point ICP with color")
        reg_p2p_color = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, threshold, trans_init
        )

        draw_registration_result_original_color(source_down, target_down, reg_p2p_color.transformation)

        draw_registration_result_original_color(pcd_mesh, pcd_gt_mesh,  reg_p2p_color.transformation)

        print("\nApply point-to-point ICP without color")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_mesh, pcd_gt_mesh, threshold, reg_p2p_color.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria
        )

        draw_registration_result_original_color(pcd_mesh, pcd_gt_mesh, reg_p2p.transformation)

    else:
        draw_registration_result_original_color(pcd_mesh, pcd_gt_mesh,trans_init)

        print("\nApply point-to-point ICP without color")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_mesh, pcd_gt_mesh, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria
        )
        draw_registration_result_original_color(pcd_mesh, pcd_gt_mesh, reg_p2p.transformation)

    return reg_p2p.transformation


def get_gt_mesh(file):
    # load gt mesh
    mesh_id = int(file.split("_")[0])
    gt_file_path = "%s/obj_%06d.ply" % (gt_meshes, mesh_id)
    # gt_tex_path = "%s/obj_%06d.png" % (gt_meshes, mesh_id)
    # print("gt_file_path: ", gt_file_path)
    # img = cv2.imread(gt_tex_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.flip(img, 0)
    gt_mesh = o3d.io.read_triangle_mesh(gt_file_path, True)
    # gt_mesh.textures = [o3d.geometry.Image(img)]
    print(gt_mesh.triangle_material_ids)

    return gt_mesh


# def remove_unwanted_triangle(pcd):
#     print(pcd.triangle_material_ids)
#
#     remove = []
#     keep = []
#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = mesh.vertices
#     # pcd.colors = o3d.utility.Vector3dVector(np.array(mesh.vertex_colors) * 1.8)
#     # pcd.normals = mesh.vertex_normals
#
#     for index, triangle in enumerate(np.asarray(pcd.triangles)):
#         pcd.triangle_material_ids = o3d.utility.IntVector(pcd.triangle_material_ids[index])
#
#         # # print(triangle)
#         # t1, t2, t3 = triangle
#         # v1 = pcd.vertices[t1]
#         # v2 = pcd.vertices[t2]
#         # v3 = pcd.vertices[t3]
#         #
#         # if v1[0] < -20 or v1[0] > 20 or v1[1] < -20 or v1[1] > 20:
#         #     remove.append(index)
#         # else:
#         #     keep.append(pcd.triangle_material_ids[index])
#
#     # pcd.remove_triangles_by_index(remove)
#     # pcd.remove_unreferenced_vertices()
#     # pcd.triangle_material_ids = o3d.utility.IntVector(keep)  # this needs to be added.
#     o3d.visualization.draw_geometries([pcd])


def get_input_data(original_mesh_path, aligned_mesh_path, method, file):
    box = o3d.geometry.OrientedBoundingBox()
    box2 = o3d.geometry.OrientedBoundingBox()
    crop_box = box.get_axis_aligned_bounding_box()
    crop_box2 = box2.get_axis_aligned_bounding_box()
    gt_mesh = get_gt_mesh(file)
    if method == "colmap":
        crop_box.min_bound = [-300, -300, -300.0]
        crop_box.max_bound = [300, 300, -4]

        crop_box2.min_bound = [-300, -300, -300.0]
        crop_box2.max_bound = [300, 300, -2.5]

        # prepare input file path
        file_path = os.path.join(original_mesh_path, file)
        print("\nLoad reconstructed mesh from %s" % file_path)
        mesh = o3d.io.read_triangle_mesh(file_path, True)

        # prepare output file path
        aligned_mesh_dir = os.path.join(aligned_mesh_path, file[:-4])
        scale = 1.0

        return aligned_mesh_dir, scale ,crop_box, crop_box2,  mesh , gt_mesh

    if method == "NGP":
        multiple = 2
        crop_box.min_bound = [-1.0 * multiple, -1.0 * multiple, -1.0 * multiple]
        crop_box.max_bound = [1.0 * multiple, -0.1 * multiple, 1.0 * multiple]

        crop_box2.min_bound = [-1.0 * multiple, -1.0 * multiple, -1.0 * multiple]
        crop_box2.max_bound = [1.0 * multiple, 0 * multiple, 1.0 * multiple]

        # prepare input file path
        file_path = os.path.join(original_mesh_path, file)
        print("\nLoad reconstructed mesh from %s" % file_path)
        mesh = o3d.io.read_triangle_mesh(file_path, True)

        # prepare output file path
        aligned_mesh_dir = os.path.join(aligned_mesh_path, file[:-4])
        scale = 300/4

        return aligned_mesh_dir, scale, crop_box, crop_box2, mesh , gt_mesh

    else:
        crop_box.min_bound = [-1,-1,-1]
        crop_box.max_bound = [1, 1, 1]
        crop_box2.min_bound = [-1,-1,-1]
        crop_box2.max_bound = [1, 1, 1]
        scale = (300 * 0.77)

        # prepare input file path
        file_path = os.path.join(original_mesh_path, file, 'mesh.obj')
        tex_path = os.path.join(original_mesh_path, file, 'material_0.png')

        img = cv2.imread(tex_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)

        mesh = o3d.io.read_triangle_mesh(file_path, True)
        mesh.textures = [o3d.geometry.Image(img)]

        o3d.visualization.draw_geometries([mesh])

        print("\nLoad reconstructed mesh from %s" % file_path)

        # prepare output file path
        aligned_mesh_dir = os.path.join(aligned_mesh_path, file)

        return aligned_mesh_dir, scale, crop_box, crop_box2, mesh , gt_mesh

def scale_mesh(method):
    print("\nMETHOD: %s" % method)

    method_mesh_path = os.path.join(reconstructed_mesh, method)
    original_mesh_path = os.path.join(method_mesh_path, 'original')
    print("original_mesh directorty: %s" % original_mesh_path)

    # prepare output filepath
    aligned_mesh_path = os.path.join(method_mesh_path, 'aligned_mesh')

    if not os.path.exists(aligned_mesh_path):
        os.mkdir(aligned_mesh_path)
    print("Saving meshes to %s" % aligned_mesh_path)

    rec_meshes = sorted(os.listdir(original_mesh_path))
    print("\nrec_meshes:")
    print(rec_meshes)
    for file in rec_meshes:
        # load the raw reconstructed mesh
        print("processing file : ", file)

        aligned_mesh_dir, scale, crop_box, crop_box2, mesh, gt_mesh = get_input_data(original_mesh_path,aligned_mesh_path, method,file)

        # crop the reconstructed mesh
        mesh_for_alignment = mesh.crop(crop_box)
        mesh_for_crop_with_gt_mesh = mesh.crop(crop_box2)

        # scale the mesh to match the gt
        mesh_for_alignment.scale(scale, center=(0, 0, 0))
        mesh_for_crop_with_gt_mesh.scale(scale, center=(0, 0, 0))

        transformation = apply_transformation(method, file, mesh_for_alignment, gt_mesh)

        ### transform the mesh_for_crop_with_gt_mesh
        mesh_for_crop_with_gt_mesh.transform(transformation)
        # mesh_final = remove_outlier_with_estimated_bbox_gt(mesh_for_crop_with_gt_mesh,gt_mesh,method)
        # mesh_final = remove_small_floating_part(mesh_final)
        # mesh_final = remove_unwanted_triangle(mesh_for_crop_with_gt_mesh)

        if method in ['colmap', 'NGP']:
            mesh_final = remove_floating_part(mesh_final)

        if not os.path.exists(aligned_mesh_dir):
            os.mkdir(aligned_mesh_dir)
        aligned_file_path = os.path.join(aligned_mesh_dir,'mesh.obj')

        print("Saving scaled mesh to %s" % aligned_file_path)
        o3d.io.write_triangle_mesh(mesh=mesh_final, filename=aligned_file_path,
                                   write_triangle_uvs=True, write_vertex_normals=True,
                                   write_vertex_colors=True)

if __name__ == '__main__':

    method = ['colmap', 'NGP']
    scale_mesh(method[2])