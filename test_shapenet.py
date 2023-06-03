import argparse
import numpy as np
import random
import os, sys
import time
import json
import csv
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt
import visdom
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from utils import *
from models import Network_No_2D_Local as Network
# from models_elements import Network_1_shot as Network
from dataset_shapenet_multi_view_silhouette import ShapeNet, collate_fn
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.ops import iterative_closest_point
from loss import get_ssim, get_iou
from PyTorchEMD.emd import earth_mover_distance
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
distChamfer = ChamferDistance()

date = time.strftime('%Y-%m-%d',time.localtime())
# from evaluation import val

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()
with open('/home/xianghui_yang/data/id2cat.json', 'r') as file:
    idx2cat = json.load(file)
    cat2idx = {}
    for idx in idx2cat.keys():
        cat2idx[idx2cat[idx]]=idx

## training param
parser.add_argument('--resume', type=str, default=None, help='optional resume model path')
parser.add_argument('--save_dir', type=str, default='out/%s_shapenet'%date, help='save directory')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimizer')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr_step', type=int, default=200, help='step to decrease lr')
parser.add_argument('--gpu', type=int, default=0, help='which gpu is available')
parser.add_argument('--target_class', nargs='+', default=["bench", "bus", "cabinet", "lamp", "pistol", "sofa", "train", "watercraft"], type=str, help='target class')

## dataset param
parser.add_argument('--root_dir_train', type=str, default='./data/Pix3D/', help='training dataset directory')
parser.add_argument('--annot_train', type=str, default='pix3d.json', help='training dataset annotation file')
parser.add_argument('--fine_tune', action='store_true', help='whether to exclude novel classes during training')
parser.add_argument('--keypoint', action='store_true', help='use only samples with keypoint annotations')

## method param
parser.add_argument('--input_size', type=int, default=224, help='input image dimension')
parser.add_argument('--img_feature_dim', type=int, default=1024, help='feature dimension for images')
parser.add_argument('--point_num', type=int, default=2500, help='number of points used in each sample')
parser.add_argument('--shape_feature_dim', type=int, default=512, help='feature dimension for shapes')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')
parser.add_argument('--smooth', type=float, default=0.2, help='activate label smoothing in classification')

opt = parser.parse_args()
print(opt)


# train_cats = ['bed']
# test_cats = ['bed']
# print('train set:', train_cats)
# print('test set:', test_cats)


model = Network(deform_num=3)
device = torch.device("cuda:0")
# model.cuda()
model = model.to(device)
# result_path = os.path.join(os.getcwd(), opt.save_dir)
# model_file_list = os.listdir(result_path)
# model_file_list = sorted(model_file_list, key=lambda item: float(item.split('-')[-1][:-4]))
# print(model_file_list)
# print('Load ', model_file_list[0])
# model_dic=torch.load(os.path.join(result_path, model_file_list[0]))

model_path = '/home/xianghui_yang/viewer_centered/log/2022-03-14_0shot_viewer_no_distance_no_2dlocal/final-model.pth'
model_dic = torch.load(model_path)
print(model_path)
model.load_state_dict(model_dic, strict=True)
model.eval()

opt.target_class += ["car","chair","monitor","plane", "rifle","speaker","table","telephone"]
dataset_test = ShapeNet(train=False, target_class=opt.target_class, input_size=opt.input_size, point_num=opt.point_num)
test_loader = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False, collate_fn=collate_fn)
print('test data consist of {} samples {} images'.format(len(dataset_test), len(dataset_test)*24))

# =================== DEFINE TRAIN ========================= #
val_distance = {}
val_normal = {}
val_init_pc_distance = {}
val_pose = {}
val_ssim = {}
val_multi_view = {}
val_emd = {}
val_fscore_tau = {}
val_fscore_2tau = {}
for cat in opt.target_class:
    val_distance[cat] = AverageValueMeter()
    val_normal[cat] = AverageValueMeter()
    val_init_pc_distance[cat] = AverageValueMeter()
    val_pose[cat] = AverageValueMeter()
    val_ssim[cat] = AverageValueMeter()
    val_multi_view[cat] = AverageValueMeter()
    val_emd[cat] = AverageValueMeter()
    val_fscore_tau[cat] = AverageValueMeter()
    val_fscore_2tau[cat] = AverageValueMeter()

with torch.no_grad():
    for _, data in enumerate(test_loader):
        trg_im, trg_mask, trg_pc, trg_normals, src_meshes, gt_pose_params, scale, centroid, fn_list, trg_cat_list, gt_multi_view = data
        trg_im = trg_im.to(device=device)
        trg_pc = trg_pc.to(device=device)
        # src_meshes = src_meshes.to(device=device)
        trg_normals = trg_normals.to(device=device)
        src_meshes = ico_sphere(4, trg_im.device).extend(trg_im.shape[0])
        gt_pose_params = gt_pose_params.to(device=device)
        scale = scale.to(device=device)
        centroid = centroid.to(device=device)
        gt_multi_view = gt_multi_view.to(device=device)

        # object centered
        # R, T = look_at_view_transform(dist=gt_pose_params[:, 2], elev=gt_pose_params[:, 1], azim=gt_pose_params[:, 0])
        # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=gt_pose_params[:, 3])
        # # transform = cameras.get_full_projection_transform()
        # raw_centroid = centroid

        # veiwer centered
        R_gt, T_gt = look_at_view_transform(dist=torch.ones(trg_im.shape[0]).cuda()*1.38, elev=gt_pose_params[:, 1], azim=gt_pose_params[:, 0])
        cameras_gt = FoVPerspectiveCameras(device=device, R=R_gt, T=torch.zeros_like(T_gt).cuda(), fov=gt_pose_params[:, 3])
        transform_gt = cameras_gt.get_world_to_view_transform()
        transform_gt_inverse = transform_gt.inverse()
        # trg_pc = transform_gt.transform_points(trg_pc).detach()
        rotate_centroid = transform_gt.transform_points(centroid.unsqueeze(1)).detach()
        rotate_centroid = rotate_centroid.squeeze()
        
        cameras = FoVPerspectiveCameras(device=device, R=torch.eye(3).unsqueeze(0).expand(trg_im.shape[0], 3, 3), T=T_gt, fov=gt_pose_params[:, 3])

        pred_meshes_list, pred_init_pcs, _ = model(trg_im, src_meshes, cameras, scale, rotate_centroid)
        # pred_meshes_list, _ = model(trg_im, cameras, scale, centroid)
        pred_meshes = pred_meshes_list[-1]
        pred_verts_canonical = transform_gt_inverse.transform_points(pred_meshes.verts_padded())
        pred_meshes = pred_meshes.update_padded(pred_verts_canonical)

        # tmp_pc = sample_points_from_meshes(pred_meshes, 30000, return_normals=False)
        # tmp_centroid = torch.mean(tmp_pc, dim=1)
        # tmp_pc = tmp_pc - tmp_centroid.unsqueeze(1).expand_as(tmp_pc)
        # tmp_scale = torch.max(torch.sqrt(torch.sum(tmp_pc ** 2, dim=2)), dim=1)[0]
        # pred_verts_canonical = pred_meshes.verts_padded()
        # pred_verts_canonical = pred_verts_canonical - tmp_centroid.unsqueeze(1).expand_as(pred_verts_canonical)
        # pred_verts_canonical = pred_verts_canonical / tmp_scale.unsqueeze(1).unsqueeze(1).expand_as(pred_verts_canonical)
        # pred_meshes = pred_meshes.update_padded(pred_verts_canonical)
        
        # pred_pc = sample_points_from_meshes(pred_meshes, opt.point_num)
        # pred_verts = pred_meshes.verts_padded()
        # icp = iterative_closest_point(pred_verts, trg_pc)
        # transformed_pred_verts = icp.Xt
        # transform_tuple = icp.RTs
        # print(transform_tuple.s, transform_tuple.R.shape, transform_tuple.T.shape)
        # pred_meshes = pred_meshes.update_padded(transformed_pred_verts)

        pred_pc, pred_normals = sample_points_from_meshes(pred_meshes, opt.point_num, return_normals=True)
        
        pred_multi_view_silhouette = get_multi_view_silhouatte_test(distance=gt_pose_params[:, 4], view=gt_pose_params[:, 3], mesh=pred_meshes, raw_scale=scale, raw_centroid=centroid)
        # verts_rgb = torch.ones_like(pred_meshes.verts_padded())
        # textures = TexturesVertex(verts_features=verts_rgb.to(device))
        # pred_meshes.textures=textures
        # _, pred_rgb = get_render_img(azim=gt_pose_params[:, 0], elev=gt_pose_params[:, 1], distance=gt_pose_params[:, 2], view=gt_pose_params[:, 3], 
        #                             mesh=pred_meshes, multi_view_distance=gt_pose_params[:, 4] , raw_scale=scale, raw_centroid=centroid, render_rgb=True)
        
        multi_view_iou = get_iou(pred_multi_view_silhouette, gt_multi_view, size_average=False).cpu()
        # print(multi_view_iou, multi_view_iou.shape)
        # pred_pose = get_pred_angle(pose_params)
        # pred_pose = pred_pose.cpu()
        # pred_pose = pred_pose*np.pi/180
        # gt_pose_params = gt_pose_params.cpu()*np.pi/180
        # cosine_similarity = torch.acos(torch.cos(pred_pose[:, 1])*torch.cos(gt_pose_params[:, 1])*torch.cos(pred_pose[:, 0]-gt_pose_params[:, 0]) + 
        #                         torch.sin(pred_pose[:, 1])*torch.sin(gt_pose_params[:, 1]))
        # cosine_similarity = cosine_similarity/np.pi*180
        # print(cosine_similarity.shape)
        # trg_im = trg_im.cpu().numpy().transpose([0,2,3,1])
        # pred_rgb = pred_rgb.cpu().numpy()
        # grey_gt = np.dot(np.array(trg_im)[..., :3], [0.2989, 0.5870, 0.1140])
        # grey_pred = np.dot(np.array(pred_rgb)[..., :3], [0.2989, 0.5870, 0.1140])

        val_emd_loss = earth_mover_distance(pred_pc, trg_pc, transpose=False)
        
        d1, d2, _, _= distChamfer(trg_pc, pred_pc)
        val_fscore = f_score(d1, d2, [0.001, 0.002])
        for batch_i in range(len(trg_cat_list)):
            val_pc_chamfer_loss, val_normal_loss = chamfer_distance(pred_pc[None, batch_i, :, :], trg_pc[None, batch_i, :, :], x_normals=pred_normals[None, batch_i, :, :], y_normals=trg_normals[None, batch_i, :, :])
            val_distance[trg_cat_list[batch_i]].update(val_pc_chamfer_loss.item())
            val_normal[trg_cat_list[batch_i]].update(val_normal_loss.item())
            # val_init_pc_chamfer_loss, _ = chamfer_distance(pred_init_pcs[None, batch_i, :, :], trg_pc[None, batch_i, :, :])
            # val_init_pc_distance[trg_cat_list[batch_i]].update(val_init_pc_chamfer_loss.item())
            # val_pose[trg_cat_list[batch_i]].update(cosine_similarity[batch_i].item())
            # val_ssim[trg_cat_list[batch_i]].update(ssim(grey_gt[batch_i], grey_pred[batch_i]))
            val_multi_view[trg_cat_list[batch_i]].update(multi_view_iou[batch_i].item())
            val_emd[trg_cat_list[batch_i]].update(val_emd_loss[batch_i].item())
            
            val_fscore_tau[trg_cat_list[batch_i]].update(val_fscore[batch_i, 0].item()) 
            val_fscore_2tau[trg_cat_list[batch_i]].update(val_fscore[batch_i, 1].item()) 
            # _verts, _faces = pred_meshes.get_mesh_verts_faces(batch_i)
            # if not os.path.exists(os.path.join('/home/xianghui_yang/object_center/log/2021-09-09_src2pcs2mesh_both_sampling_multi_view_0_shot', trg_cat_list[batch_i])):
            #     os.mkdir(os.path.join('/home/xianghui_yang/object_center/log/2021-09-09_src2pcs2mesh_both_sampling_multi_view_0_shot', trg_cat_list[batch_i]))
            # save_obj(os.path.join('/home/xianghui_yang/object_center/log/2021-09-09_src2pcs2mesh_both_sampling_multi_view_0_shot', trg_cat_list[batch_i], fn_list[batch_i]+'.obj'), _verts, _faces)

mean_chamfer = []
mean_normal = []
mean_init_chamfer = []
mean_degree = []
# mean_ssim = []
mean_multi_view_silhouette = []
mean_emd = []
mean_f_score_tau = []
mean_f_score_2tau = []
# with open('test_results.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Class", "CD", "EMD", "IoU"])

print("Class", "Chamfer", "Normal",  "EMD", "Silhouette", "F-score tau", "F-score 2tau")
for cat in opt.target_class:
    mean_chamfer.append(val_distance[cat].avg*1000)
    mean_normal.append(val_normal[cat].avg*10)
    mean_init_chamfer.append(val_init_pc_distance[cat].avg)
    mean_degree.append(val_pose[cat].avg)
    # mean_ssim.append(val_ssim[cat].avg)
    mean_multi_view_silhouette.append(val_multi_view[cat].avg*100)
    mean_emd.append(val_emd[cat].avg*100)
    mean_f_score_tau.append(val_fscore_tau[cat].avg)
    mean_f_score_2tau.append(val_fscore_2tau[cat].avg)
    print(cat, "%.6f"%(val_distance[cat].avg*1000), "%.6f"%(val_normal[cat].avg*10), "%.5f"%(val_emd[cat].avg/opt.point_num*100), "%.5f"%(val_multi_view[cat].avg*100), "%.2f"%(val_fscore_tau[cat].avg), "%.2f"%(val_fscore_2tau[cat].avg))
        
    # writer.writerow([cat, val_distance[cat].avg, val_emd[cat].avg/opt.point_num, val_multi_view[cat].avg])

    

print('mean mesh pc', np.mean(mean_chamfer))
print('mean mesh normal', np.mean(mean_normal))
print('mean init pc', np.mean(mean_init_chamfer))
print('mean degree', np.mean(mean_degree))
# print('mean ssim', np.mean(mean_ssim))
print('mean multi view', np.mean(mean_multi_view_silhouette))
print('mean EMD:', np.mean(mean_emd)/opt.point_num)
print('mean f-score 1e-3', np.mean(mean_f_score_tau))
print('mean f-score 2e-3', np.mean(mean_f_score_2tau))