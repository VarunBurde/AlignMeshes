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
texture_mesh = os.path.join(colmap_mesh_path, 'texture_mesh')

if not os.path.exists(texture_mesh):
    os.mkdir(texture_mesh)

def main():
    rec_meshes = sorted(os.listdir(aligned_mesh_path))
    print("\nrec_meshes")
    print(rec_meshes)
    for file in rec_meshes:
        print("processing file : ", file)
        ms = pymeshlab.MeshSet()
        input_file_path = os.path.join(aligned_mesh_path, file)
        ms.load_new_mesh(input_file_path)
        ms.generate_surface_reconstruction_screened_poisson()
        # ms.generate_surface_reconstruction_vcg()
        # ms.generate_surface_reconstruction_ball_pivoting()
        # not familiar with the crop API, but I'm sure it's doable
        # now we generate UV map; there are a couple options here but this is a naive way
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
        # create texture using UV map and vertex colors

        output_object_dir = os.path.join(texture_mesh, file[:-4])
        if not os.path.exists(output_object_dir):
            os.mkdir(output_object_dir)
        # texture file won't be saved until you save the mesh
        ms.compute_texmap_from_color(textname=f"material_0.png")
        ms.save_current_mesh(os.path.join(output_object_dir, 'mesh.ply'), save_face_color=True, save_textures=True)





if __name__ == '__main__':
    main()