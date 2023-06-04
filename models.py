import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import resnet
import numpy as np
from pytorch3d.ops import vert_align, sample_points_from_meshes, GraphConv, packed_to_padded, padded_to_packed, SubdivideMeshes
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import join_meshes_as_batch, Pointclouds
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.renderer import TexturesVertex
# from pointnet2_utils import PointNetSetAbstraction
from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule, my_PointnetSAModule
from pytorch3d.renderer import (
    look_at_view_transform,
    TexturesVertex,
    FoVOrthographicCameras, 
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)
import sys
sys.path.append("./external/")
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
chamfer_distance = ChamferDistance()

subdivide = SubdivideMeshes()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class PointNet2(nn.Module):
#     def __init__(self, bottleneck_size, normal_channel=False):
#         super(PointNet2, self).__init__()
#         in_channel = 6 if normal_channel else 3
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
#         self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
#         self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
#         self.fc1 = nn.Linear(1024, 1024)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(1024, bottleneck_size)
#         self.bn2 = nn.BatchNorm1d(bottleneck_size)
#         self.drop2 = nn.Dropout(0.4)

#     def forward(self, xyz):
#         B, _, _ = xyz.shape
#         if self.normal_channel:
#             norm = xyz[:, 3:, :]
#             xyz = xyz[:, :3, :]
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         x = l3_points.view(B, 1024)
#         x = self.drop1(F.relu(self.bn1(self.fc1(x))))
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         return x

class PointNetpp(nn.Module):
    def __init__(self, bottleneck_size=512):
        super(PointNetpp, self).__init__()
        self.sa1 = my_PointnetSAModule(npoint=512, radius=0.2, nsample=32, mlp=[0, 64, 64, 128], use_xyz=True)
        self.sa2 = my_PointnetSAModule(npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 128, 256], use_xyz=True)
        self.sa3 = my_PointnetSAModule(mlp=[256, 256, 256, 512], use_xyz=True)
        # self.sa3 = my_PointnetSAModule(npoint=32, radius=0.8, nsample=128, mlp=[256, 256, 256, 512], use_xyz=True)
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(True),
        #     nn.Dropout(0.4),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, bottleneck_size),
        # )
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, bottleneck_size)
        self.bn2 = nn.BatchNorm1d(bottleneck_size)
        # self.drop2 = nn.Dropout(0.4)

    def forward(self, xyz):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # xyz, features = self._break_up_pc(pointcloud)
        xyz, features = self.sa1(xyz, None)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa3(xyz, features)
        
        # return self.fc_layer(features.squeeze(-1))
        x = F.relu(self.bn1(self.fc1(features.squeeze(-1))))
        x = F.relu(self.bn2(self.fc2(x)))
        return x

class PointDeform(nn.Module):
    def __init__(self, bottleneck_size=2500, output_dim=3):
        super(PointDeform, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.th(self.conv4(x))
        return x

class Network_0_shot(nn.Module):
    def __init__(self, bottleneck_size=512, deform_num=3):
        super(Network_0_shot, self).__init__()
        self.img_encoder = resnet.resnet18(num_classes=bottleneck_size)
        self.pc_encoder = PointNetpp(bottleneck_size=bottleneck_size)
        self.deform_num = deform_num
        self.deform_layers = PointDeform(bottleneck_size=3 + 128+256+512)
        self.sphere_deform_layers = nn.ModuleList([PointDeform(bottleneck_size=3 + 128+256+512 + 128+256+512) for _ in range(self.deform_num)])
        self.img_feature_project = nn.ModuleList([
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        ])
        self.pc_feature_project = nn.ModuleList([
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        ])
    
    def sample_img_features(self, points, transform, img_feats, raw_scale, raw_centroid):
        points = points * raw_scale.unsqueeze(1) + raw_centroid.unsqueeze(1)
        points_pos = transform.transform_points(points)
        # flip y coordinate
        device, dtype = points_pos.device, points_pos.dtype
        factor = torch.tensor([1, -1, 1], device=device, dtype=dtype).view(1, 1, 3)
        points_pos = points_pos * factor
        points_align_feats = vert_align(img_feats, points_pos)

        return points_align_feats
    
    def sample_pointnetpp_features(self, points, points_features):
        points_align_feats = []
        for xyz, feats in points_features:
            feats = feats.transpose(1,2)
            if xyz==None:
                points_align_feats.append(feats.expand(-1, points.shape[1], -1))
            else:
                _, _, idx, _ = chamfer_distance(points, xyz)
                idx = idx.detach()
                idx = idx.long().unsqueeze(2).expand(-1, -1, feats.shape[-1])
                sampled_feats = feats.gather(1, idx)
                points_align_feats.append(sampled_feats)

        points_align_feats = torch.cat(points_align_feats, dim=2)
        return points_align_feats

    def forward(self, im, src_mesh, cameras, scale, centroid):
        device = im.device
        batch_size = im.shape[0]
        img_feature_map = []
        pcs_feature_list = []
        img_feature = self.img_encoder.conv1(im)
        img_feature = self.img_encoder.bn1(img_feature)
        img_feature = self.img_encoder.relu(img_feature)
        img_feature = self.img_encoder.maxpool(img_feature)
        img_feature = self.img_encoder.layer1(img_feature)
        img_feature = self.img_encoder.layer2(img_feature)
        img_feature_map.append(self.img_feature_project[0](img_feature))
        img_feature = self.img_encoder.layer3(img_feature)
        img_feature_map.append(self.img_feature_project[1](img_feature))
        img_feature = self.img_encoder.layer4(img_feature)
        img_feature_map.append(self.img_feature_project[2](img_feature))

        src_pcs = sample_points_from_meshes(src_mesh, 2500)
        src_pcs = src_pcs.detach()
        transform = cameras.get_full_projection_transform()
        img_feature_grid = self.sample_img_features(src_pcs, transform, img_feature_map, scale, centroid)

        deform_feature = torch.cat((src_pcs, img_feature_grid), 2).transpose(1,2)
        deform_pcs = self.deform_layers(deform_feature)
        deform_pcs = deform_pcs.transpose(1,2).contiguous()
        pred_pcs = src_pcs + deform_pcs
        pcs_feature_list = []
        l1_xyz, l1_points = self.pc_encoder.sa1(pred_pcs.detach(), None)
        pcs_feature_list.append((l1_xyz, self.pc_feature_project[0](l1_points)))
        l2_xyz, l2_points = self.pc_encoder.sa2(l1_xyz, l1_points)
        pcs_feature_list.append((l2_xyz, self.pc_feature_project[1](l2_points)))
        l3_xyz, l3_points = self.pc_encoder.sa3(l2_xyz, l2_points)
        pcs_feature_list.append((l3_xyz, self.pc_feature_project[2](l3_points)))

        pred_meshes_list = []
        move_list = []
        # verts_num_list = [162, 642, 2562]
        for i in range(self.deform_num):
            if i==0:
                pred_mesh = ico_sphere(2, device).extend(im.shape[0])
            else:
                pred_mesh = subdivide(pred_mesh)
            vertices = pred_mesh.verts_padded()
            pc_feature_grid = self.sample_pointnetpp_features(vertices, pcs_feature_list)
            img_feature_grid = self.sample_img_features(vertices, transform, img_feature_map, scale, centroid)
            deform_feature = torch.cat((vertices, pc_feature_grid, img_feature_grid), 2).transpose(1,2).contiguous()

            offsets = self.sphere_deform_layers[i](deform_feature)
            offsets = offsets.transpose(1,2).contiguous()
            pred_mesh = pred_mesh.offset_verts(offsets.view(-1, 3))
            pred_meshes_list.append(pred_mesh)
            if self.training:
                move_list.append(torch.mean(torch.sum(torch.pow(offsets, 2), 1)))
                
        return pred_meshes_list, pred_pcs, move_list


class Network_for_Pix3D(nn.Module):
    def __init__(self, bottleneck_size=512, deform_num=3):
        super(Network_for_Pix3D, self).__init__()
        self.img_encoder = resnet.resnet18(num_classes=bottleneck_size)
        self.pc_encoder = PointNetpp(bottleneck_size=bottleneck_size)
        self.deform_num = deform_num
        self.deform_layers = PointDeform(bottleneck_size=3 + 128+256+512)
        self.sphere_deform_layers = nn.ModuleList([PointDeform(bottleneck_size=3 + 128+256+512 + 128+256+512) for i in range(0, self.deform_num)])
        self.img_feature_project = nn.ModuleList([
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        ])
        self.pc_feature_project = nn.ModuleList([
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        ])
    
    def sample_img_features(self, points, raw_scale, raw_centroid, transform, f_pix3d, img_feats):
        points = points * raw_scale.unsqueeze(1) + raw_centroid.unsqueeze(1)
        points_pos = transform.transform_points(points)
        # flip y coordinate
        device, dtype = points_pos.device, points_pos.dtype
        f = f_pix3d*2/32
        # camera coordinate system
        # print(points_pos.shape, f.shape)
        # points_pos
        points_pos = f * points_pos / points_pos[:, :, 2, None]
        factor = torch.tensor([-1, -1, 1], device=device, dtype=dtype).view(1, 1, 3)
        points_pos = points_pos * factor
        points_align_feats = vert_align(img_feats, points_pos)

        return points_align_feats
    
    def sample_pointnetpp_features(self, points, points_features):
        points_align_feats = []
        for xyz, feats in points_features:
            feats = feats.transpose(1,2)
            if xyz==None:
                points_align_feats.append(feats.expand(-1, points.shape[1], -1))
            else:
                _, _, idx, _ = chamfer_distance(points, xyz)
                idx = idx.detach()
                idx = idx.long().unsqueeze(2).expand(-1, -1, feats.shape[-1])
                sampled_feats = feats.gather(1, idx)
                points_align_feats.append(sampled_feats)

        points_align_feats = torch.cat(points_align_feats, dim=2)
        return points_align_feats

    def forward(self, im, src_mesh, transform, scale, centroid, f_pix3d):
        device = im.device
        batch_size = im.shape[0]
        img_feature_map = []
        pcs_feature_list = []
        img_feature = self.img_encoder.conv1(im)
        img_feature = self.img_encoder.bn1(img_feature)
        img_feature = self.img_encoder.relu(img_feature)
        img_feature = self.img_encoder.maxpool(img_feature)
        img_feature = self.img_encoder.layer1(img_feature)
        img_feature = self.img_encoder.layer2(img_feature)
        img_feature_map.append(self.img_feature_project[0](img_feature))
        img_feature = self.img_encoder.layer3(img_feature)
        img_feature_map.append(self.img_feature_project[1](img_feature))
        img_feature = self.img_encoder.layer4(img_feature)
        img_feature_map.append(self.img_feature_project[2](img_feature))

        src_pcs = sample_points_from_meshes(src_mesh, 2500)
        src_pcs = src_pcs.detach()
        img_feature_grid = self.sample_img_features(src_pcs, scale, centroid, transform, f_pix3d, img_feature_map)

        deform_feature = torch.cat((src_pcs, img_feature_grid), 2).transpose(1,2)
        deform_pcs = self.deform_layers(deform_feature)
        deform_pcs = deform_pcs.transpose(1,2).contiguous()
        pred_pcs = src_pcs + deform_pcs

        pcs_feature_list = []
        l1_xyz, l1_points = self.pc_encoder.sa1(pred_pcs.detach(), None)
        pcs_feature_list.append((l1_xyz, self.pc_feature_project[0](l1_points)))
        l2_xyz, l2_points = self.pc_encoder.sa2(l1_xyz, l1_points)
        pcs_feature_list.append((l2_xyz, self.pc_feature_project[1](l2_points)))
        l3_xyz, l3_points = self.pc_encoder.sa3(l2_xyz, l2_points)
        pcs_feature_list.append((l3_xyz, self.pc_feature_project[2](l3_points)))

        pred_meshes_list = []
        move_list = []
        # verts_num_list = [162, 642, 2562]
        for i in range(self.deform_num):
            if i==0:
                pred_mesh = ico_sphere(2, device).extend(im.shape[0])
            else:
                pred_mesh = subdivide(pred_mesh)
            vertices = pred_mesh.verts_padded()
            pc_feature_grid = self.sample_pointnetpp_features(vertices, pcs_feature_list)
            # pc_feature_grid = trg_feature.unsqueeze(1).expand(trg_feature.shape[0], vertices.shape[1], 896)
            img_feature_grid = self.sample_img_features(vertices, scale, centroid, transform, f_pix3d, img_feature_map)
            deform_feature = torch.cat((vertices, pc_feature_grid, img_feature_grid), 2).transpose(1,2)
            offsets = self.sphere_deform_layers[i](deform_feature)
            offsets = offsets.transpose(1,2).contiguous()
            pred_mesh = pred_mesh.offset_verts(offsets.view(-1, 3))
            pred_meshes_list.append(pred_mesh)
            if self.training:
                move_list.append(torch.mean(torch.sum(torch.pow(offsets, 2), 1)))
                
        return pred_meshes_list, pred_pcs, move_list

if __name__ == "__main__":
    import time
    # import torch.optim as optim
    from collections import OrderedDict
    from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
    # from utils import get_render_img
    # from loss import get_iou_loss

    device = torch.device("cuda:0")
    img = torch.rand(16, 3, 224, 224).cuda()
    trg_mask = torch.ones(32, 224, 224).cuda()
    gt_pc = torch.rand(16, 2500, 3).cuda()
    gt_pose_params = torch.zeros(16, 2).cuda()
    distance = torch.ones(16).cuda()
    view = torch.ones(16).cuda()*50
    scale = torch.ones(16, 1).cuda()
    centroid = torch.zeros(16, 3).cuda()
    R, T = look_at_view_transform(dist=distance, elev=gt_pose_params[:, 1], azim=gt_pose_params[:, 0])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view)
    # P = cameras.get_projection_transform().get_matrix()
    transform = cameras.get_full_projection_transform()

    meshes = ico_sphere(4, img.device).extend(img.shape[0])
    print(meshes.verts_padded().shape)

    
    model = Network_0_shot(deform_num=3)
    model = model.to(img.device)
    pred_meshes_list, pred_pcs, offset_list = model(img, meshes, transform, scale, centroid)
    print(pred_meshes_list[0].verts_padded().shape, pred_meshes_list[1].verts_padded().shape, pred_meshes_list[2].verts_padded().shape)
    print(pred_pcs.shape)
    print(offset_list[0], offset_list[1], offset_list[2])
