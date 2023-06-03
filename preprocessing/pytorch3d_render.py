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

def pytorch3d_render(param):
    catid = param[0]
    objid = param[1]
    save_path = os.path.join("/home/xianghui_yang/data/ShapeNet/Pytorch3D_Render", catid, objid)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    elif len(os.listdir(save_path))==24:
        return
    render_path = os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetRendering", catid, objid)
    if not os.path.exists(render_path):
        render_path = os.path.join("/home/xianghui_yang/data/ShapeNet/my_ShapeNetRendering", catid, objid)
    # mat = sio.loadmat(os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetV1PointCloud_Normalized", catid, objid+'.mat'))
    # raw_scale = torch.from_numpy(mat['scale'])
    # raw_centroid = torch.from_numpy(mat['centroid'])*torch.FloatTensor([[1,1,-1]])
    
    verts, faces, _  = load_obj(os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetCore.v1", catid, objid, 'model.obj'), load_textures=False)
    verts_corrected = torch.cat([verts[:, 2, None], verts[:, 1, None], verts[:, 0, None]], 1)
    verts_rgb = torch.ones_like(verts_corrected).unsqueeze(0)
    textures = TexturesVertex(verts_features=verts_rgb)
    try:
        mesh = Meshes(verts=[verts_corrected], faces=[faces.verts_idx], textures=textures).cuda()
    except:
        print('mesh io error', catid, objid)
        return
    device = mesh.device
    distance = []
    azim = []
    elev = []
    view = []
    with open(os.path.join(render_path, 'rendering/rendering_metadata.txt'),'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            param = line.split(' ')
            azim.append(float(param[0]))
            elev.append(float(param[1]))
            distance.append(float(param[3])*1.75)
            view.append(float(param[4])*2)

    mesh = mesh.extend(24)
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
    # print(R, T)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    sigma = 1e-4
    raster_settings = RasterizationSettings(image_size=137, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=50, perspective_correct=False)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftSilhouetteShader())
    try:
        silhouette = renderer_silhouette(mesh, cameras=cameras, lights=lights)[..., 3] # B*H*W
        silhouette = torch.flip(silhouette, dims=[2]).detach()
        renderer_rgb = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftPhongShader(device=device, cameras=cameras, lights=lights))
        im_rgb = renderer_rgb(mesh, cameras=cameras, lights=lights)[..., :3] # B*H*W*3
        im_rgb = torch.flip(im_rgb, dims=[2])
        render_result = im_rgb*(silhouette.unsqueeze(-1)).detach()
        render_result = np.uint8(render_result.cpu().numpy()*255)
        for i in range(24):
            pil_image = Image.fromarray(render_result[i])
            pil_image.save(os.path.join(save_path, "%02d.jpg"%i))
    except:
        traceback.print_exc()
        print(mesh.verts_padded().shape, mesh.faces_padded().shape)
        print(catid, objid)
    return


def pytorch3d_silhouette_render(param):
    catid = param[0]
    objid = param[1]
    save_path = os.path.join("/home/xianghui_yang/data/ShapeNet/Pytorch3D_Silhouette_Render", catid, objid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # elif len(os.listdir(save_path))==16:
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
    

    with open(os.path.join(render_path, 'rendering/rendering_metadata.txt'),'r') as f:
        lines = f.readlines()
        line = lines[0].rstrip()
        cam_param = line.split(' ')
        distance = torch.FloatTensor([float(cam_param[3])*1.75]).to(device)
        view = torch.FloatTensor([float(cam_param[4])*2]).to(device)

    render_num = 8
    mesh = mesh.extend(render_num)
    # x = torch.FloatTensor([45, -45]).to(device)
    # y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    x = torch.FloatTensor([0]).to(device)
    y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    elevation, azimuth = torch.meshgrid(x, y)
    elevation = elevation.reshape(render_num)
    azimuth = azimuth.reshape(render_num)
    distance = distance.expand(render_num)
    view = view.expand(render_num)
    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    # print(R, T)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
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
            pil_image.save(os.path.join(save_path, "%02d_%02d_%03d.jpg"%(16+i, elevation[i], azimuth[i])))
    except:
        traceback.print_exc()
        print(mesh.verts_padded().shape, mesh.faces_padded().shape)
        print(catid, objid)
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

with open('/home/xianghui_yang/data/id2cat.json', 'r') as file:
    idx2cat = json.load(file)
cat2idx = {}
for idx in idx2cat.keys():
    cat2idx[idx2cat[idx]]=idx

cat_list = ["plane","car","chair","monitor","telephone","speaker","table", "rifle", "bench", "bus", "cabinet", "lamp", "pistol", "sofa", "train", "watercraft"]
# cat_list = ["car","chair","monitor","plane", "rifle","speaker","table","telephone"]

for cat in cat_list:
    print(cat)
    catid = cat2idx[cat]
    if catid in os.listdir("/home/xianghui_yang/data/ShapeNet/my_ShapeNetRendering"):
        objid = os.listdir(os.path.join("/home/xianghui_yang/data/ShapeNet/my_ShapeNetRendering", catid))
    else:
        objid = os.listdir(os.path.join("/home/xianghui_yang/data/ShapeNet/ShapeNetRendering", catid))
    # if not os.path.exists(os.path.join("/home/xianghui_yang/data/ShapeNet/Pytorch3D_Render", catid)):
    #     os.mkdir(os.path.join("/home/xianghui_yang/data/ShapeNet/Pytorch3D_Render", catid))
    params = [(catid, item) for item in objid]
    # with Pool(2) as p:
    #     p.map(pytorch3d_render, params)
    for param in params:
        pytorch3d_silhouette_render(param)

# import time
# start = time.time()
# # os.mkdir(os.path.join("/home/xianghui_yang/data/ShapeNet/Pytorch3D_Depth_Render", '02818832'))
# # # for i in range(10):
# pytorch3d_silhouette_render(('02691156', '1c93b0eb9c313f5d9a6e43b878d5b335'))
# print(time.time()-start)