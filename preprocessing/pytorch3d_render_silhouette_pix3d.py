import os
import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import padded_to_packed
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, TexturesVertex
from PIL import Image
from multiprocessing import Pool
import scipy.io as sio
import json
import traceback



def pytorch3d_silhouette_render(cat, modelid, model_path, distance):
    save_path = os.path.join("/home/xianghui_yang/data/Pix3D/Pytorch3D_Silhouette_Render", cat, modelid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # mat = sio.loadmat(os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetV1PointCloud_Normalized", catid, objid+'.mat'))
    # raw_scale = torch.from_numpy(mat['scale'])
    # raw_centroid = torch.from_numpy(mat['centroid'])*torch.FloatTensor([[1,1,-1]])
    
    verts, faces, _  = load_obj(os.path.join("/home/xianghui_yang/data/Pix3D", model_path), load_textures=False)
    # verts_corrected = torch.cat([verts[:, 2, None], verts[:, 1, None], verts[:, 0, None]], 1)
    verts_corrected = verts*torch.FloatTensor([[1,1,-1]])
    device = torch.device('cuda:0')
    try:
        mesh = Meshes(verts=[verts_corrected], faces=[faces.verts_idx]).to(device)
    except:
        print('mesh io error', cat, modelid)
        return
    

    # with open(os.path.join(render_path, 'rendering/rendering_metadata.txt'),'r') as f:
    #     lines = f.readlines()
    #     line = lines[0].rstrip()
    #     cam_param = line.split(' ')
    #     distance = torch.FloatTensor([float(cam_param[3])*1.75]).to(device)
    #     view = torch.FloatTensor([float(cam_param[4])*2]).to(device)

    render_num = 24
    mesh = mesh.extend(render_num)
    x = torch.FloatTensor([45, 0, -45]).to(device)
    y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    # x = torch.FloatTensor([0]).to(device)
    # y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    elevation, azimuth = torch.meshgrid(x, y)
    elevation = elevation.reshape(render_num)
    azimuth = azimuth.reshape(render_num)
    distance = distance.expand(render_num)
    # view = view.expand(render_num)
    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    # print(R, T)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    lights = PointLights(location=[[0.0, 0.0, -5.0]], device=device)
    sigma = 1e-5
    raster_settings = RasterizationSettings(image_size=224, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=50, perspective_correct=False, max_faces_per_bin=50000)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftSilhouetteShader())
    try:
        silhouette = renderer_silhouette(mesh, cameras=cameras, lights=lights)[..., 3] # B*H*W
        silhouette = torch.flip(silhouette, dims=[2]).detach()
        silhouette = np.uint8(silhouette.cpu().numpy()*255)
        for i in range(render_num):
            pil_image = Image.fromarray(silhouette[i])
            pil_image.save(os.path.join(save_path, "%02d_%02d_%03d.jpg"%(i, elevation[i], azimuth[i])))
    except:
        traceback.print_exc()
        print(mesh.verts_padded().shape, mesh.faces_padded().shape)
        print(cat, modelid)
    return


def pytorch3d_depth_render(param):
    catid = param[0]
    objid = param[1]
    save_path = os.path.join("/home/xianghui_yang/data/ShapeNet/Pytorch3D_Depth_Render", catid, objid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # elif len(os.listdir(save_path))==24:
    #     return
    render_path = os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetRendering", catid, objid)
    if not os.path.exists(render_path):
        render_path = os.path.join("/home/xianghui_yang/data/ShapeNet/my_ShapeNetRendering", catid, objid)
    # mat = sio.loadmat(os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetV1PointCloud_Normalized", catid, objid+'.mat'))
    # raw_scale = torch.from_numpy(mat['scale'])
    # raw_centroid = torch.from_numpy(mat['centroid'])*torch.FloatTensor([[1,1,-1]])
    
    verts, faces, _  = load_obj(os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetCore.v1", catid, objid, 'model.obj'), load_textures=False)
    verts_corrected = torch.cat([verts[:, 2, None], verts[:, 1, None], verts[:, 0, None]], 1)
    device = torch.device('cuda:0')
    try:
        mesh = Meshes(verts=[verts_corrected], faces=[faces.verts_idx]).to(device)
    except:
        print('mesh io error', catid, objid)
        return

    azimuth_list = []
    elevation_list = []
    distance_list = []
    view_list = []
    with open(os.path.join(render_path, 'rendering/rendering_metadata.txt'),'r') as f:
        lines = f.readlines()
        for line in lines:
            cam_param = line.rstrip().split(' ')
            azimuth_list.append(float(cam_param[0])) 
            elevation_list.append(float(cam_param[1])) 
            distance_list.append(float(cam_param[3])*1.75)
            view_list.append(float(cam_param[4])*2)
    
    

    azimuth = torch.FloatTensor(azimuth_list).to(device)
    elevation = torch.FloatTensor(elevation_list).to(device)
    distance = torch.FloatTensor(distance_list).to(device)
    view = torch.FloatTensor(view_list).to(device)
    mesh = mesh.extend(24)

    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    # print(R, T)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    sigma = 1e-4
    
    raster_settings = RasterizationSettings(image_size=224, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=1, perspective_correct=False, max_faces_per_bin=50000)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    try:
        fragments = rasterizer(mesh)
        depth = fragments.zbuf[:, :, :, 0]
        depth = torch.flip(depth, dims=[2]).detach()
        norm_depth = depth.view(depth.shape[0], -1)
        depth_max = torch.max(norm_depth, dim=1)[0].unsqueeze(1).unsqueeze(1)
        depth_min = torch.min(norm_depth, dim=1)[0].unsqueeze(1).unsqueeze(1)
        # print(depth.shape, depth_min.shape, depth_max.shape)
        depth = (depth - depth_min)/(depth_max-depth_min)
        # depth = np.uint16(depth.cpu().numpy()*255)
        depth = depth.cpu().numpy()
        # print(depth.shape)
        for i in range(depth.shape[0]):
        #     pil_image = Image.fromarray(depth[i])
        #     # pil_image = Image.fromarray(depth[i], mode='I;16').convert(mode='I')
        #     pil_image.save(os.path.join(save_path, "%02d.png"%(i)))
            np.save(os.path.join(save_path, "%02d.npy"%(i)), depth[i])

        
    except:
        traceback.print_exc()
        print(mesh.verts_padded().shape, mesh.faces_padded().shape)
        print(catid, objid)
    return


cat_list = ['bed', 'bookcase', 'chair', 'desk', 'sofa', 'table', 'tool', 'wardrobe', 'misc']

with open('/home/xianghui_yang/data/Pix3D/pix3d.json', 'r') as file:
    pix3d = json.load(file)


# for meta_info in pix3d:
#     if meta_info['category'] in cat_list:
#         img_path = os.path.join('/home/xianghui_yang/data/Pix3D', meta_info['img'])
#         mask_path = os.path.join('/home/xianghui_yang/data/Pix3D', meta_info['mask'])
#         gt_model_path = os.path.join('/home/xianghui_yang/data/Pix3D', meta_info['model'])
        # save_path = os.path.join('/home/xianghui_yang/object_center/real_img/', meta_info['category'])
        
        # R = meta_info['rot_mat']
        # T = meta_info['trans_mat']
        # f_pix3d = meta_info['focal_length']
        # f_pix3d = torch.FloatTensor([f_pix3d]).unsqueeze(0).to(device)
        # R_1 = torch.FloatTensor([[1,0,0],[0,1,0],[0,0,-1]])
        # R = torch.FloatTensor(R)
        # T = torch.FloatTensor(T)


for cat in cat_list:
    model_dir = os.path.join('/home/xianghui_yang/data/Pix3D/model', cat)
    for model_id in os.listdir(model_dir):
        if cat=='chair' and model_id in ('IKEA_SKRUVSTA', 'IKEA_SNILLE_1', 'IKEA_JULES_1', 'IKEA_PATRIK', 'IKEA_MARKUS'):
            continue
        # model_id = meta_info['model'].split('/')[-2]
        gt_model_path = os.path.join(model_dir, model_id, 'model.obj')
        distance = torch.FloatTensor([1.])
        print(cat, model_id, gt_model_path, distance)

        pytorch3d_silhouette_render(cat, model_id, gt_model_path, distance)
        # exit()

# import time
# start = time.time()
# # os.mkdir(os.path.join("/home/xianghui_yang/data/ShapeNet/Pytorch3D_Depth_Render", '02818832'))
# # # for i in range(10):
# pytorch3d_silhouette_render(('02691156', '1c93b0eb9c313f5d9a6e43b878d5b335'))
# print(time.time()-start)