import os
import scipy.io as sio
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import TexturesAtlas
from multiprocessing import Pool
from pytorch3d.io import load_obj

MAT_DIR = '/home/xianghui_yang/data/ShapeNet/ShapeNetV1PointCloud_Normalized_RGB/'
if not os.path.exists(MAT_DIR):
    os.mkdir(MAT_DIR)
# cat_idx_list = os.listdir('/home/xianghui_yang/data/ShapeNet/ShapeNetRendering')
# cat_idx_list.remove('rendering_only.tgz')

def generate_pc_normal(obj_file_param):
    cat_idx = obj_file_param[0]
    obj_file = obj_file_param[1]

    # trg_mat = sio.loadmat('/home/xianghui_yang/data/ShapeNet/CleanShapeNet_v1_mat/'+ cat_idx +'/' + obj_file)
    # trg_verts = torch.from_numpy(trg_mat['v'])
    # trg_faces_idx = torch.from_numpy(trg_mat['f'])
    # trg_mesh = Meshes(verts=[trg_verts], faces=[trg_faces_idx])
    try:
        verts, faces, aux = load_obj('/home/xianghui_yang/data/ShapeNet/ShapeNetCore.v1/'+ cat_idx +'/'+obj_file+'/model.obj', load_textures=True, create_texture_atlas=True, texture_atlas_size=4)
        verts_corrected = torch.cat([verts[:, 2, None], verts[:, 1, None], -verts[:, 0, None]], 1)
        faces_idx = faces.verts_idx
        # print(aux.texture_atlas.shape)
        # print(verts_corrected.dtype)
        textures = TexturesAtlas(atlas=[aux.texture_atlas])
        trg_mesh = Meshes(verts=[verts_corrected], faces=[faces_idx], textures=textures)
        trg_pc, trg_normal, trg_rgb = sample_points_from_meshes(trg_mesh, num_samples=30000, return_normals=True, return_textures=True)
        trg_pc = trg_pc[0].numpy()
        centroid = np.mean(trg_pc, axis=0)
        trg_pc = trg_pc - centroid
        scale = np.max(np.sqrt(np.sum(trg_pc ** 2, axis=1)))
        trg_pc = trg_pc / scale
        trg_normal = trg_normal[0].numpy()
        trg_rgb = trg_rgb[0].numpy()
        # print(centroid, scale)
        # print(trg_pc.shape, trg_normal.shape, trg_pc.dtype, trg_normal.dtype)

        verts_corrected = verts_corrected.numpy()
        verts_corrected = verts_corrected - centroid
        verts_corrected = verts_corrected / scale

        new_file_path = MAT_DIR + cat_idx +'/'+ obj_file + '.mat'
        sio.savemat(new_file_path, {'pc': trg_pc, 'n':trg_normal, "rgb":trg_rgb, 'verts':verts_corrected,  'faces':faces_idx.numpy(), 'centroid':centroid, 'scale':scale})
    except:
        print(cat_idx, obj_file)


# 'plane': '02691156', ** 
# 'bench': '02828884', **
# 'cabinet': '02933112', **
# 'car': '02958343', **
# 'chair': '03001627', **
# 'monitor': '03211117', **
# 'lamp': '03636649', **
# 'speaker': '03691459', **
# 'firearm': '04090263', **
# 'couch': '04256520',  **
# 'table': '04379243', **
# 'cellphone': '04401088', **
# 'watercraft': '04530566'  **

# 03211117 04401088
# print(cat_idx_list)

cat_idx_list = ["04090263","04401088"]

for cat_idx in cat_idx_list:
    print(cat_idx)
    if not os.path.exists(MAT_DIR + cat_idx+'/'):
        os.mkdir(MAT_DIR + cat_idx+'/')
    # obj_list = os.listdir('/home/xianghui_yang/data/ShapeNet/CleanShapeNet_v1_mat/'+ cat_idx)
    obj_list = os.listdir('/home/xianghui_yang/data/ShapeNet/ShapeNetCore.v1/'+ cat_idx)
    cat_name_list = [(cat_idx, item) for item in obj_list]
    print('Files Number: ', len(cat_name_list))

    with Pool(10) as p:
        p.map(generate_pc_normal, cat_name_list)

    # generate_pc_normal(cat_name_list[0])