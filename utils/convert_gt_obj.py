import copy
import json
import os
import trimesh
import numpy as np
import open3d as o3d
import pymeshlab
from scipy.spatial.transform import Rotation
import cv2

method = ['neus', 'monosdf', 'volsdf', 'gt_method']

root_path = os.path.split(os.path.split(__file__)[0])[0]
mesh_path = os.path.join(root_path, 'meshes')
gt_meshes = os.path.join(os.path.join(mesh_path, 'gt_mesh'))
gt_meshes_obj = os.path.join(os.path.join(mesh_path, 'gt_mesh_obj'))

if not os.path.exists(gt_meshes_obj):
    os.mkdir(gt_meshes_obj)
reconstructed_mesh = os.path.join(mesh_path, 'reconstructed_mesh')

for file in os.listdir(gt_meshes):
    if file[-4:] == '.ply':
        gt_file_location = os.path.join(gt_meshes,file)
        gt_png_file = os.path.join(gt_meshes,file[:-4] +'.png')

        gt_file_output = os.path.join(gt_meshes_obj,file[:-4] +'.ply')


        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(gt_file_location)
        ms.convert_pervertex_uv_into_perwedge_uv()
        ms.transfer_texture_to_vertex_color_1_or_2_meshes()
        ms.save_current_mesh(gt_file_output)



