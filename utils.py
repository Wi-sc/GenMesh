import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pytorch3d.ops import padded_to_packed
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    camera_position_from_spherical_angles,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)
import torchvision.transforms.functional as tf
import emd_cuda

class MeshRendererWithDepth(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments.zbuf[..., 0]

def resize_padding(im, desired_size, mode="RGB"):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new(mode, (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def KaiMingInit(net):
    """Kaiming Init layer parameters."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2)  # slope = 0.2 in the original implementation
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

def remove_nan(net):
    """Kaiming Init layer parameters."""
    for m in net.children():
        print(m)
        if not torch.isfinite(m.weight).all():
            print("Nan from weights")
        if m.bias is not None and not torch.isfinite(m.bias).all():
            print("Nan from bias")
        if isinstance(m, torch.nn.Conv2d):
            if not torch.isfinite(m.weight).all():
                torch.nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None and not torch.isfinite(m.bias).all():
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv1d):
            if not torch.isfinite(m.weight).all():
                torch.nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None and not torch.isfinite(m.bias).all():
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            if not torch.isfinite(m.weight).all():
                torch.nn.init.constant_(m.weight, 1)
            if not torch.isfinite(m.bias).all():
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm1d):
            if not torch.isfinite(m.weight).all():
                torch.nn.init.constant_(m.weight, 1)
            if not torch.isfinite(m.bias).all():
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            if not torch.isfinite(m.weight).all():
                torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None and not torch.isfinite(m.bias).all():
                torch.nn.init.constant_(m.bias, 0)
    return net


def load_checkpoint(model, pth_file):
    """load state and network weights"""
    checkpoint = torch.load(pth_file, map_location=lambda storage, loc: storage.cuda())
    if 'model' in checkpoint.keys():
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Previous weight loaded')


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.avg * (self.count / (self.count + n)) + val * (n / (self.count + n))
        self.count += n


def get_pred_from_cls_output(outputs):
    preds = []
    for n in range(len(outputs)):
        output = outputs[n]
        _, pred = output.topk(1, 1, True, True)
        preds.append(pred.view(-1))
    return preds


def accuracy(outputs, targets):
    """Compute accuracy for each euler angle separately"""
    with torch.no_grad():  # no grad computation to reduce memory
        preds = get_pred_from_cls_output(outputs)
        res = []
        for n in range(0, len(outputs)):
            res.append(100. * torch.mean((preds[n] == targets[:, n]).float()))
        return res


def angles_to_matrix(angles, is_vector=True):
    """Compute the rotation matrix from euler angles for a mini-batch"""
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) - torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) + torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)
    element4 = (-torch.cos(rol) * torch.sin(azi) - torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element5 = (-torch.sin(rol) * torch.sin(azi) + torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)

    if is_vector:
        return torch.cat((element1, element2, element3, element4, element5, element6, element7, element8, element9), dim=1)
    else:
        col_1 = torch.cat((element1, element2, element3), dim=1).unsqueeze(2)
        col_2 = torch.cat((element4, element5, element6), dim=1).unsqueeze(2)
        col_3 = torch.cat((element7, element8, element9), dim=1).unsqueeze(2)
        return torch.cat((col_1, col_2, col_3), dim=2)


def rotation_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    preds = preds.float().clone()
    targets = targets.float().clone()
    preds[:, 1] = preds[:, 1] - 180.
    preds[:, 2] = preds[:, 2] - 180.
    targets[:, 1] = targets[:, 1] - 180.
    targets[:, 2] = targets[:, 2] - 180.
    preds = preds * np.pi / 180.
    targets = targets * np.pi / 180.
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi
    return R_err


def rotation_acc(preds, targets, th=30.):
    R_err = rotation_err(preds, targets)
    return 100. * torch.mean((R_err <= th).float())


def angle_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    errs = torch.abs(preds - targets)
    errs = torch.min(errs, 360. - errs)
    return errs

def get_pred_angle(azim_ele, bin_size=15):
    _, azi_cls = azim_ele[0].topk(1, 1, True, True)
    azi_cls = azi_cls.view(-1)
    pred_azi_delta = azim_ele[1]
    delta_value = pred_azi_delta[torch.arange(pred_azi_delta.size(0)), azi_cls.long()].tanh() / 2
    pred_azi = (azi_cls.float() + delta_value + 0.5) * bin_size
    pred_ele = (azim_ele[2] + 0.5) * 5 + 25
    return torch.cat([pred_azi.unsqueeze(1), pred_ele], dim=1)
    

    


def calculate_area(vertices, faces):
    num_batch = faces.shape[0]
    num_faces = faces.shape[1]
    v_1 = torch.gather(vertices, dim=1, index=faces[:, :, 0].unsqueeze(-1).expand(num_batch, num_faces, 3))
    v_2 = torch.gather(vertices, dim=1, index=faces[:, :, 1].unsqueeze(-1).expand(num_batch, num_faces, 3))
    v_3 = torch.gather(vertices, dim=1, index=faces[:, :, 2].unsqueeze(-1).expand(num_batch, num_faces, 3))
    face_areas = torch.norm(torch.cross(v_2-v_1, v_3-v_1), dim=2)/2
    return face_areas


def get_render_img(azim, elev, distance, view, mesh, raw_centroid, raw_scale):
    device = azim.device
    R, T = look_at_view_transform(dist=distance, elev=elev, azim=azim)
    # print(R, T)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    sigma = 1e-4
    raster_settings = RasterizationSettings(image_size=224, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=50, perspective_correct=False)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftSilhouetteShader())
    
    
    mesh = mesh.scale_verts(raw_scale)
    raw_centroid = raw_centroid.unsqueeze(1).expand_as(mesh.verts_padded())
    mesh = mesh.offset_verts(padded_to_packed(raw_centroid, mesh.mesh_to_verts_packed_first_idx(), mesh.verts_padded_to_packed_idx().shape[0]))

    silhouette = renderer_silhouette(mesh, cameras=cameras, lights=lights) 
    silhouette = silhouette[..., 3] # B*H*W
    silhouette = torch.flip(silhouette, dims=[2])
    # depth_map = torch.flip(depth_map, dims=[2])

    renderer_rgb = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), 
                                shader=SoftPhongShader(device=device, cameras=cameras, lights=lights))
    im_rgb = renderer_rgb(mesh, cameras=cameras, lights=lights)[..., :3] # B*H*W*3
    im_rgb = torch.flip(im_rgb, dims=[2])

    return silhouette, im_rgb*(silhouette.unsqueeze(-1)).detach()

def get_multi_view_silhouatte(mesh, raw_centroid, raw_scale, distance, view):
    mesh = mesh.scale_verts(raw_scale)
    raw_centroid = raw_centroid.unsqueeze(1).expand_as(mesh.verts_padded())
    mesh = mesh.offset_verts(padded_to_packed(raw_centroid, mesh.mesh_to_verts_packed_first_idx(), mesh.verts_padded_to_packed_idx().shape[0]))

    device = mesh.device
    batch_size = len(mesh)
    x = torch.FloatTensor([45, -45, 0]).to(device)
    y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    elevation, azimuth = torch.meshgrid(x, y)
    elevation = elevation.reshape(1, 24).expand(batch_size, 24).contiguous()
    azimuth = azimuth.reshape(1, 24).expand(batch_size, 24).contiguous()
    distance = distance.unsqueeze(1).expand(batch_size, 24).contiguous()
    view = view.unsqueeze(1).expand(batch_size, 24).contiguous()

    random_view_id = np.random.choice(range(24), 8, replace=False)
    elevation = elevation[:, random_view_id].view(-1)
    azimuth = azimuth[:, random_view_id].view(-1)
    distance = distance[:, random_view_id].view(-1)
    view = view[:, random_view_id].view(-1)

    mesh = mesh.extend(8) # [m_1*8, m_2*8, ..., m_n*8]
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    sigma = 1e-4
    raster_settings = RasterizationSettings(image_size=224, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=50, perspective_correct=False)
    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftSilhouetteShader())
    silhouette = renderer_silhouette(mesh, cameras=cameras, lights=lights)[..., 3] # B*H*W
    silhouette = torch.flip(silhouette, dims=[2])
    return silhouette.view(batch_size, 8, 224, 224), random_view_id

def get_nmr_mutli_view_silhouatte(mesh, R):
    device = mesh.device
    batch_size = len(mesh)
    view_per_model = R.shape[1]
    mesh = mesh.extend(view_per_model) # [m_1*8, m_2*8, ..., m_n*8]
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    sigma = 1e-4
    T_gt = torch.zeros(batch_size*view_per_model, 3)
    T_gt[..., 2] = 2.732
    fov_defult = torch.ones(batch_size*view_per_model)*30.0
    R = R.view(batch_size*view_per_model, 3, 3)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T_gt, fov=fov_defult)
    raster_settings = RasterizationSettings(image_size=224, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=50, perspective_correct=False)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftSilhouetteShader())
    silhouette = renderer_silhouette(mesh, cameras=cameras, lights=lights)[..., 3] # B*H*W
    silhouette = torch.flip(silhouette, dims=[1, 2])
    return silhouette.view(batch_size, view_per_model, 224, 224)

def get_element_silhouatte(mesh, raw_centroid, raw_scale, distance, view):
    """
        mesh: [B*8]
        raw_scale: [B, 1]
        raw_centroid: [B, 3]
        distance: [B]
        view: [B]
    """
    device = mesh.device
    batch_size = len(mesh)
    element_num = 8
    real_batch_size = batch_size//element_num
    view_num = 4
    
    raw_scale = raw_scale.unsqueeze(1).expand(real_batch_size, element_num, -1).reshape(real_batch_size*element_num, 1)
    raw_centroid = raw_centroid.unsqueeze(1).expand(real_batch_size, element_num, 3).reshape(real_batch_size*element_num, 3)
    distance = distance.unsqueeze(1).expand(real_batch_size, element_num*view_num).contiguous().view(-1)
    view = view.unsqueeze(1).expand(real_batch_size, element_num*view_num).contiguous().view(-1)

    mesh = mesh.scale_verts(raw_scale)
    raw_centroid = raw_centroid.unsqueeze(1).expand_as(mesh.verts_padded())
    mesh = mesh.offset_verts(padded_to_packed(raw_centroid, mesh.mesh_to_verts_packed_first_idx(), mesh.verts_padded_to_packed_idx().shape[0]))

    
    # x = torch.FloatTensor([45, -45, 0]).to(device)
    # y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    x = torch.FloatTensor([45, -45, 0]).to(device)
    y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    elevation, azimuth = torch.meshgrid(x, y)
    elevation = elevation.reshape(1, 24).expand(batch_size, 24).contiguous()
    azimuth = azimuth.reshape(1, 24).expand(batch_size, 24).contiguous()
    

    random_view_id = np.random.choice(range(24), view_num, replace=False)
    elevation = elevation[:, random_view_id].view(-1)
    azimuth = azimuth[:, random_view_id].view(-1)
    # distance = distance[:, random_view_id].view(-1)
    # view = view[:, random_view_id].view(-1)

    mesh = mesh.extend(view_num) # [m_1*8, m_2*8, ..., m_n*8]
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    sigma = 1e-4
    raster_settings = RasterizationSettings(image_size=128, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=20, perspective_correct=False)
    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftSilhouetteShader())
    silhouette = renderer_silhouette(mesh, cameras=cameras, lights=lights)[..., 3] # B*H*W
    silhouette = torch.flip(silhouette, dims=[2])
    return silhouette.view(batch_size, view_num, 128, 128)


def get_multi_view_silhouatte_test(mesh, raw_centroid, raw_scale, distance, view):
    mesh = mesh.scale_verts(raw_scale)
    raw_centroid = raw_centroid.unsqueeze(1).expand_as(mesh.verts_padded())
    mesh = mesh.offset_verts(padded_to_packed(raw_centroid, mesh.mesh_to_verts_packed_first_idx(), mesh.verts_padded_to_packed_idx().shape[0]))
    device = mesh.device
    batch_size = len(mesh)
    
    x = torch.FloatTensor([45, -45, 0]).to(device)
    y = torch.FloatTensor([0, 45, 90, 135, 180, 225, 270, 315]).to(device)
    elevation, azimuth = torch.meshgrid(x, y)
    elevation = elevation.reshape(1, 24).expand(batch_size, 24).contiguous().view(-1)
    azimuth = azimuth.reshape(1, 24).expand(batch_size, 24).contiguous().view(-1)
    distance = distance.unsqueeze(1).expand(batch_size, 24).contiguous().view(-1)
    view = view.unsqueeze(1).expand(batch_size, 24).contiguous().view(-1)

    mesh = mesh.extend(24) # [m_1*24, m_2*24, ..., m_n*24]
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    sigma = 1e-4
    raster_settings = RasterizationSettings(image_size=224, blur_radius=np.log(1. / 1e-4 - 1.)*sigma, faces_per_pixel=50, perspective_correct=False)
    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftSilhouetteShader())
    silhouette = renderer_silhouette(mesh, cameras=cameras, lights=lights)[..., 3] # B*H*W
    silhouette = torch.flip(silhouette, dims=[2])
    return silhouette.view(batch_size, 24, 224, 224)

def get_points_feature_map(points, point_features, raw_scale, raw_centroid, cameras, size, radius):
    device = points.device
    batch_size = points.shape[0]
    point_features = point_features.transpose(1,2).contiguous()
    points = points * raw_scale.unsqueeze(1) + raw_centroid.unsqueeze(1)

    point_cloud = Pointclouds(points=points, features=point_features)
    raster_settings = PointsRasterizationSettings(image_size=size, radius=radius, points_per_pixel = 50)
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    feature_maps = renderer(point_cloud)
    feature_maps = torch.flip(feature_maps, dims=[2])
    return feature_maps.view(batch_size, size*size, -1)


def get_grey_img(img):
    return tf.rgb_to_grayscale(img, 1)


def project_verts(verts, P, eps=1e-1):
    """
    Project verticies using a 4x4 transformation matrix

    Inputs:
    - verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions.
    - P: FloatTensor of shape (N, 4, 4) giving projection matrices

    Outputs:
    - verts_out: FloatTensor of shape (N, V, 3) giving vertex positions (x, y, z)
        where verts_out[i] is the result of transforming verts[i] by P[i].
    """
    # Handle unbatched inputs
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device

    # Add an extra row of ones to the world-space coordinates of verts before
    # multiplying by the projection matrix. We could avoid this allocation by
    # instead multiplying by a 4x3 submatrix of the projectio matrix, then
    # adding the remaining 4x1 vector. Not sure whether there will be much
    # performance difference between the two.
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))

    # Avoid division by zero by clamping the absolute value
    w = verts_cam_hom[:, :, 3:]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)

    verts_proj = verts_cam_hom[:, :, :3] / w

    if singleton:
        return verts_proj[0]
    return verts_proj

def get_blender_intrinsic_matrix(N=None):
    """
    This is the (default) matrix that blender uses to map from camera coordinates
    to normalized device coordinates. We can extract it from Blender like this:

    import bpy
    camera = bpy.data.objects['Camera']
    render = bpy.context.scene.render
    K = camera.calc_matrix_camera(
         render.resolution_x,
         render.resolution_y,
         render.pixel_aspect_x,
         render.pixel_aspect_y)
    """
    K = [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
    K = torch.tensor(K)
    if N is not None:
        K = K.view(1, 4, 4).expand(N, 4, 4)
    return K

def get_volume(mesh):

    batch_size = len(mesh)
    normal = mesh.faces_normals_padded()
    face_area = mesh.faces_areas_packed().view(batch_size, -1)

    v_p = mesh.verts_packed()
    f_p = mesh.faces_packed()
    f_1 = v_p[f_p[:, 0]].view(batch_size, -1, 3)
    h = torch.matmul(f_1.unsqueeze(2), normal.unsqueeze(-1)).squeeze() # [b, f, 3] [b, f, 3] => 

    volume = h*face_area/3
    return volume.sum(1)

def f_score(dist_label, dist_pred, threshold):
    batch_size = dist_label.shape[0]
    num_threshold = len(threshold)
    num_label = dist_label.shape[1]
    num_predict = dist_pred.shape[1]

    num_label = torch.tensor([num_label]*batch_size).to(dist_pred.device)
    num_predict = torch.tensor([num_predict]*batch_size).to(dist_pred.device)
    f_scores = torch.zeros((batch_size, num_threshold)).to(dist_pred.device)
    
    for i in range(num_threshold):
        num = torch.where(dist_label <= threshold[i], 1, 0).sum(1)
        
        recall = 100.0 * num / num_label
        num = torch.where(dist_pred <= threshold[i], 1, 0).sum(1)
        
        precision = 100.0 * num / num_predict
		# f_scores.append((2*precision*recall)/(precision+recall+1e-8))
        f_scores[:, i] = (2*precision*recall)/(precision+recall+1e-8)
    return f_scores


def get_face_center(vertices, faces):
    num_batch = vertices.shape[0]
    num_faces = faces.shape[1]
    v_1 = torch.gather(vertices, dim=1, index=faces[:, :, 0].unsqueeze(-1).expand(num_batch, num_faces, 3))
    v_2 = torch.gather(vertices, dim=1, index=faces[:, :, 1].unsqueeze(-1).expand(num_batch, num_faces, 3))
    v_3 = torch.gather(vertices, dim=1, index=faces[:, :, 2].unsqueeze(-1).expand(num_batch, num_faces, 3))
    return (v_1 + v_2 + v_3)/3

def match(v1, v2):
    match_id = emd_cuda.approxmatch_forward(v1, v2)
    match_id = match_id.detach().type(torch.int64)
    print(match)
    return match_id


def merge_elements(v_1, v_2, f, threshold=0.01):
    batch_size = v_1.shape[0]
    device = v_1.device
    f_center_1 = get_face_center(v_1, f)
    f_center_2 = get_face_center(v_2, f)
    dist, id1, _, _ = chamfer(f_center_1, f_center_2)
    _, face_to_merge_1  = torch.min(dist, 1)
    face_to_merge_2 = id1[face_to_merge_1]

    v1_to_merge = v_1[face_to_merge_1]
    v2_to_merge = v_2[face_to_merge_2]
    match_id = match(v1_to_merge, v2_to_merge) # [B, N1, N2]
    v_1[face_to_merge_1] = v2_to_merge[match_id]
    
    f1 = f.clone()
    f2 = f.clone()
    f1[face_to_merge_1] = [-1, -1, -1]
    f2[face_to_merge_1] = [-1, -1, -1]

    return v1, v2, f1, f2