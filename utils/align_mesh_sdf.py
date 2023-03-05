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
#root_path = "/home/abenbihi/ws/tf/sdfstudio/outputs/shared_data/Clean_meshes/"
mesh_path = os.path.join(root_path, 'meshes')
# mesh_path = os.path.join(os.path.split(root_path)[0], 'meshes')
gt_meshes = os.path.join(os.path.join(mesh_path, 'gt_mesh'))
reconstructed_mesh = os.path.join(mesh_path, 'reconstructed_mesh')
parameter_file = os.path.join(root_path,'params')

DEBUG = (0==1)

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
        #'05_mustard_bottle',
        #'06_tuna_fish_can',
        #'07_pudding_box',
        #'08_gelatin_box',
        #'09_large_clamp',
        #'10_banana',
        '17_scissors',
        #'21_foam_brick'
        ]


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
    mesh_0 = remove_floating_part(mesh_0)
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
    return mesh_1

def remove_outlier_with_estimated_bbox_gt(mesh, gt_mesh, method):
    if method == "colmap":
        aabb = gt_mesh.get_oriented_bounding_box()
        aabb.scale(1.05, aabb.get_center())
        mesh = mesh.crop(aabb)
        o3d.visualization.draw_geometries([mesh])
        return mesh
    else:
        aabb = gt_mesh.get_oriented_bounding_box()
        aabb.scale(1.2, aabb.get_center())
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



    rec_meshes = sorted(os.listdir(original_mesh_path))
    print("\nrec_meshes:")
    print(rec_meshes)
    for file in rec_meshes:
        # load the raw reconstructed mesh
        #print("processing file : ", file)
        ms = pymeshlab.MeshSet()
        scale = int(300 * 0.77)
        
        if method == "nerfacto":
            file_path = os.path.join(original_mesh_path, file, 'poisson_mesh.ply')
        else:
            file_path = os.path.join(original_mesh_path, file, 'mesh.obj')
            img_file = os.path.join(original_mesh_path, file, 'material_0.png')
            img = cv2.imread(img_file)
            mesh = o3d.io.read_triangle_mesh(file_path)
            mesh.textures = [o3d.geometry.Image(img)]
            o3d.visualization.draw_geometries([mesh])



        print("\nLoad reconstructed mesh from %s"%file_path)
        ms.load_new_mesh(file_path)

        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=50)
        per = pymeshlab.Percentage(27)
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag= per)
        ms.meshing_remove_folded_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_cut_along_crease_edges()
        # print(ms.compute_selection_by_condition_per_face())


        # process it
        #https://pymeshlab.readthedocs.io/en/2021.10/filter_list.html
        if PYMESHLAB_VERSION == "2021.10":
            ms.repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_close_holes()
            ms.matrix_set_from_translation_rotation_scale(scalex=scale, scaley=scale, scalez=scale)
        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html
        elif PYMESHLAB_VERSION == "latest":
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_close_holes()
            ms.compute_matrix_from_translation_rotation_scale(scalex=scale, scaley=scale, scalez=scale)
        else:
            raise ValueError("Unknown PYMESHLAB_VERSION")

        # save scaled mesh
        scale_mesh_file_dir = os.path.join(scaled_mesh_path, file)
        if not os.path.exists(scale_mesh_file_dir):
            os.mkdir(scale_mesh_file_dir)
        output_path = os.path.join(scale_mesh_file_dir, 'mesh.obj')
        output_path2 = os.path.join(scale_mesh_file_dir, 'mesh2.obj')
        # ms.save_current_mesh(output_path, save_face_color=True, save_textures=True)
        print("Saving scaled mesh to %s"%output_path)
        o3d.io.write_triangle_mesh(mesh=mesh, filename=output_path2,
                                   write_triangle_uvs=True, write_vertex_normals=True,
                                   write_vertex_colors= True)




def align_mesh(method):
    print("\nMETHOD: %s"%method)
    params_file = os.path.join(parameter_file, "%s_icp.json"%method)
    with open(params_file, "r") as f:
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
        scaled_meshes = nerfacto_scaled_meshes
        #scaled_meshes = volsdf_scaled_meshes

    for file in scaled_meshes:
        #print("processing file : ", file)
        # load scaled mesh
        file_path = os.path.join(scaled_mesh_path, file, 'mesh.obj')
        img =  os.path.join(scaled_mesh_path, file, 'material_0.png')
        mesh = o3d.io.read_triangle_mesh(file_path,True)
        mesh.textures = [o3d.geometry.Image(img)]

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

        # draw_registration_result(pcd_mesh, pcd_gt_mesh, trans_init)

        print("\nApply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_mesh, pcd_gt_mesh, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=criteria
        )
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # draw_registration_result(pcd_mesh, pcd_gt_mesh, reg_p2p.transformation)



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

        # if method =='nerfacto':
        #     ms.laplacian_smooth()
        
        # save aligned mesh
        align_mesh_file_dir = os.path.join(aligned_mesh_path, file)
        if not os.path.exists(align_mesh_file_dir):
            os.mkdir(align_mesh_file_dir)
        output_path = os.path.join(align_mesh_file_dir, 'mesh.obj')
        output_path2 = os.path.join(align_mesh_file_dir, 'mesh2.obj')
        texture_path = os.path.join(align_mesh_file_dir, 'material_0.png')
        ms.save_current_mesh(output_path, save_face_color=True, save_textures=True)



        # mesh2 = o3d.io.read_triangle_mesh(output_path)
        # o3d.io.write_triangle_mesh(mesh=mesh2, filename=output_path2,
        #                            write_triangle_uvs=True, write_vertex_normals=True,
        #                            write_vertex_colors= True)

        # trim = trimesh.load(output_path)
        # trimgt = trimesh.load(gt_file_path)
        #
        # box = trimesh.creation.box(extents= trimgt.bounding_box.extents)
        # trim.export(output_path2)
        # box.apply_scale(1.1)
        # trim = trim.slice_plane(box.facets_origin, -box.facets_normal)
        # trim.export(output_path2)


        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(output_path))
        # # mesh = mesh.compute_convex_hull()
        # # mesh.compute_uvatlas()
        # mesh.material.material_name = 'defaultLit'
        # mesh.material.texture_maps['albedo'] = o3d.t.io.read_image(texture_path)
        # o3d.t.io.write_triangle_mesh(mesh=mesh, filename=output_path2, write_triangle_uvs=True)


        # ms1 = pymeshlab.MeshSet()
        # ms1.add_mesh(mesh)
        # ms1.vertex_matrix()
        # box = ms.BoundingBox()
        # print(box.max())
        # mesh.transform(reg_p2p.transformation)
        # mesh = remove_outlier_with_estimated_bbox_gt(mesh, gt_mesh, method)
        # print(mesh.has_triangle_uvs())
        # o3d.io.write_triangle_mesh(mesh=mesh, filename=output_path2, write_triangle_uvs=True)

        # ms = pymeshlab.MeshSet()
        # ms.load_new_mesh(output_path2)
        # ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=2048)
        # ms.compute_texmap_from_color(textname="material_1.png", texth=2048, textw=2048)
        # ms.save_current_mesh(output_path, save_face_color=True, save_textures=True)




if __name__=="__main__":

    method = ['neus', 'volsdf', 'monosdf', 'nerfacto']

    #main(method[2])
    
    scale_mesh(method[0])
    # align_mesh(method[0])
