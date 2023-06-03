import argparse
import yaml
import numpy as np
import random
import os, sys
import time
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt
import visdom
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from models import Network_0_shot as Network
# from Pixel2Mesh import Network
# from shapenet import ShapeNet, collate_fn
from dataset_shapenet_multi_view_silhouette import ShapeNet, collate_fn
from utils import KaiMingInit, get_multi_view_silhouatte, get_grey_img, get_multi_view_silhouatte_test, AverageValueMeter
from loss import get_iou, get_iou_loss
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, save_obj
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, FoVPerspectiveCameras


sys.path.append("./external/")
from pyTorchChamferDistance.chamfer_distance import ChamferDistance
distChamfer = ChamferDistance()

date = time.strftime('%Y-%m-%d',time.localtime())

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()


## training param
parser.add_argument('--resume', type=str, default=None, help='optional resume model path')
parser.add_argument('--save_dir', type=str, default='out/%s_shapenet'%date, help='save directory')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--n_epoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--lr_step', type=int, default=100, help='step to decrease lr')
parser.add_argument('--gpu', type=str, default='0', help='which gpu is available')
parser.add_argument('--base_class', nargs='+', default=["car","chair","monitor","plane", "rifle","speaker","table","telephone"], type=str, help='target class')
parser.add_argument('--novel_class', nargs='+', default=["bench", "bus", "cabinet", "lamp", "pistol", "sofa", "train", "watercraft"], type=str, help='target class')

## dataset param
parser.add_argument('--root_dir_train', type=str, default='/home/xianghui_yang/data/ShapeNet/', help='training dataset directory')
parser.add_argument('--annot_train', type=str, default='pix3d.json', help='training dataset annotation file')
parser.add_argument('--fine_tune', action='store_true', help='whether to exclude novel classes during training')
parser.add_argument('--keypoint', action='store_true', help='use only samples with keypoint annotations')

## method param
parser.add_argument('--img_size', type=int, default=224, help='input image dimension')
parser.add_argument('--img_feature_dim', type=int, default=512, help='feature dimension for images')
parser.add_argument('--point_num', type=int, default=2500, help='number of points used in each sample')
parser.add_argument('--deform_num', type=int, default=1, help='deform layers number')

parser.add_argument('--chamfer_weight', type=float, default=1, help='chamfer distance loss weight')
parser.add_argument('--normal_weight', type=float, default=1e-3, help='chamfer normal distance loss weight')
parser.add_argument('--smooth_weight', type=float, default=1e-3, help='smooth loss weight')
parser.add_argument('--normal_consistency_weight', type=float, default=1e-2, help='normal consistency loss weight')
parser.add_argument('--edge_weight', type=float, default=0.1, help='edge length loss weight')
parser.add_argument('--pose_weight', type=float, default=0.01, help='pose loss weight')
parser.add_argument('--silhouette_weight', type=float, default=0.01, help='silhouett loss weight')

## visdom parameters
parser.add_argument('--display_port', type=int, default=8097, help='Display port')
parser.add_argument('--display_id', type=int, default=1, help='Display id')
parser.add_argument('--win_size', type=int, default=320, help='Display window size')
parser.add_argument('--name', type=str, default='ShapeNet', help='task name')
opt = parser.parse_args()
print(opt, flush=True)
print('train set:', opt.base_class)
print('test set:', opt.novel_class)
# print(opt)
parm_dict = vars(opt)
with open(opt.save_dir+'/config.yaml', 'w') as f:
    yaml.dump(parm_dict, f)



dataset_train = ShapeNet(train=True, target_class=opt.base_class, input_size=opt.img_size, point_num=opt.point_num)

dataset_val = ShapeNet(train=False, target_class=opt.base_class+opt.novel_class, input_size=opt.img_size, point_num=opt.point_num)

train_loader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=opt.workers, drop_last=False, collate_fn=collate_fn)
print('train data consist of {} samples {} images'.format(len(dataset_train), len(dataset_train)*24))
print('val data consist of {} samples {} images'.format(len(dataset_val), len(dataset_val)*24))
# ========================================================== #

# ================CREATE NETWORK============================ #

model = Network(bottleneck_size=opt.img_feature_dim, deform_num=opt.deform_num)
device = torch.device("cuda:0")
model = model.to(device)
print(model)
model.apply(KaiMingInit)

# loed_model_name = '/home/xianghui_yang/object_center/log/2021-11-03_src2pcs2mesh_0shot_iou1e-1/model-99-0.0082.pth'
# pretrained_dict = torch.load(loed_model_name)
# new_pretrained_dict = OrderedDict()
# for k,v in pretrained_dict.items():
#     new_pretrained_dict[k]=pretrained_dict[k]
#     if k.split('.')[0]=='deform_layers':
#         new_k = k.split('.')
#         new_k[0] = 'sphere_deform_layers'
#         new_pretrained_dict['.'.join(new_k)]=pretrained_dict[k]
#     if k.split('.')[0]=='pc_encoder':
#         new_pretrained_dict[k]=pretrained_dict[k]
# model.load_state_dict(new_pretrained_dict, strict=False)
# loed_model_name = '/home/xianghui_yang/few_shot_3d/out/2021-04-03_src2trg_offset_1/pretrain.pth'
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in torch.load(loed_model_name).items() if (k in model_dict)}
# for k,v in new_pretrained_dict.items():
#     print(k)
# model_dict.update(pretrained_dict)
# model.load_state_dict(pretrained_dict, strict=True)
# print("Previous weight loaded: ", loed_model_name)
# model.load_state_dict(new_pretrained_dict, strict=False)
# print("Previous weight loaded: ", loed_model_name)


# =============DEFINE stuff for logs ======================= #
result_path = os.path.join(os.getcwd(), opt.save_dir)
if not os.path.exists(result_path):
    os.makedirs(result_path)

l1_loss = nn.L1Loss()
# ========================================================== #
train_loss_avg_log = {
    'sum': AverageValueMeter(),
    'chamfer': AverageValueMeter(),
    'init chamfer': AverageValueMeter(),
    'normal': AverageValueMeter(),
    'edge': AverageValueMeter(),
    'smooth': AverageValueMeter(),
    'multi_view': AverageValueMeter()
}

assert opt.display_id > 0
vis = visdom.Visdom(server='http://172.16.13.175', port = opt.display_port, env=opt.save_dir.split('/')[-1], use_incoming_socket=False)
plot_data_train = {'X':[],'Y':[], 'legend':[item for item in train_loss_avg_log.keys()]}
plot_data_val = {'X':[],'Y':[], 'Init PCs':[], 'Pose':[], 'IoU':[], 'legend':[str(item) for item in opt.novel_class]+['mean']}
train_dataloader_size = len(train_loader)
print_freq = 100
val_dataloader_size = len(val_loader)
# display_single_pane_ncols = opt.display_single_pane_ncols

# =================== DEFINE TRAIN ========================= #
def train(data_loader, optimizer):
    for k in train_loss_avg_log.keys():
        train_loss_avg_log[k].reset()
    model.train()
    for i, data in enumerate(data_loader):
        trg_im, trg_mask, trg_pc, trg_normals, src_meshes, gt_pose_params, scale, centroid, _, _, gt_multi_view = data
        trg_im = trg_im.cuda()
        trg_mask = trg_mask.cuda()
        trg_pc = trg_pc.cuda()
        trg_normals = trg_normals.cuda()
        # src_meshes = src_meshes.cuda()
        gt_pose_params = gt_pose_params.cuda()
        scale = scale.cuda()
        centroid = centroid.cuda()
        gt_multi_view = gt_multi_view.cuda()
        src_meshes = ico_sphere(4, trg_im.device).extend(trg_im.shape[0])
        
        # trg_pc = trg_pc * scale.unsqueeze(1) + centroid.unsqueeze(1)
        
        
        R_gt, T_gt = look_at_view_transform(dist=torch.ones(opt.batch_size).cuda()*1.4, elev=gt_pose_params[:, 1], azim=gt_pose_params[:, 0])
        cameras_gt = FoVPerspectiveCameras(device=device, R=R_gt, T=torch.zeros_like(T_gt).cuda(), fov=gt_pose_params[:, 3])
        transform_gt = cameras_gt.get_world_to_view_transform()
        transform_gt_inverse = transform_gt.inverse()
        rotate_centroid = transform_gt.transform_points(centroid.unsqueeze(1)).detach()
        rotate_centroid = rotate_centroid.squeeze()
        # R, T = look_at_view_transform(dist=torch.ones(opt.batch_size).cuda()*1.38, elev=torch.zeros(opt.batch_size).cuda(), azim=torch.zeros(opt.batch_size).cuda())
        cameras = FoVPerspectiveCameras(device=device, R=torch.eye(3).unsqueeze(0).expand(opt.batch_size, 3, 3), T=T_gt, fov=gt_pose_params[:, 3])
        pred_meshes_list, pred_init_pcs, offset_list = model(trg_im, src_meshes, cameras, scale, rotate_centroid)
        # pred_meshes_list, offset_list = model(trg_im, src_meshes, cameras, scale, rotate_centroid)
        # print('pred verts', torch.any(torch.isnan(pred_meshes_list[-1].verts_padded())))
        
        pred_init_pcs = transform_gt_inverse.transform_points(pred_init_pcs)
        loss_chamfer_init_pcs, _= chamfer_distance(trg_pc, pred_init_pcs)
        # loss_chamfer_init_pcs = torch.tensor(0.).cuda()
        loss_chamfer = torch.tensor(0.).cuda()
        loss_normal = torch.tensor(0.).cuda()
        loss_edge = torch.tensor(0.).cuda()
        loss_smooth = torch.tensor(0.).cuda()
        # loss_normal_consistency = torch.tensor(0.).cuda()
        # loss_silhouette = torch.tensor(0.).cuda()

        num_meshes = len(pred_meshes_list)
        skip_iter = False
        for mesh_i in range(num_meshes):
            pred_meshes = pred_meshes_list[mesh_i]
            verts = pred_meshes.verts_padded()
            if not torch.isfinite(verts).all():
                skip_iter = True
                break
            pred_verts_canonical = transform_gt_inverse.transform_points(verts)
            pred_meshes = pred_meshes.update_padded(pred_verts_canonical)
            pred_meshes_list[mesh_i] = pred_meshes
            
            pred_pc, pred_normals = sample_points_from_meshes(pred_meshes, opt.point_num, return_normals=True)
            # pred_pc = transform_gt_inverse(pred_pc)
            # pred_normals = transform_gt_inverse(pred_normals)

            sub_loss_chamfer, sub_loss_normal = chamfer_distance(trg_pc, pred_pc, x_normals=trg_normals, y_normals=pred_normals)
            loss_chamfer = loss_chamfer + sub_loss_chamfer
            loss_normal = loss_normal + sub_loss_normal
            loss_edge = loss_edge + mesh_edge_loss(pred_meshes)
            loss_smooth = loss_smooth + offset_list[mesh_i]
        
        if skip_iter:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!Found NaN!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

        # if epoch>=100:
        #     final_meshes = pred_meshes_list[-1]
        #     # verts = final_meshes.verts_padded()
        #     # _centroid = torch.mean(verts, dim=1)
        #     # verts = verts - _centroid.unsqueeze(1).expand_as(verts)
        #     # _scale = torch.max(torch.sqrt(torch.sum(verts ** 2, dim=2)), dim=1)[0] + 1e-4
        #     # verts = verts / _scale.unsqueeze(1).unsqueeze(1)
        #     # final_meshes = final_meshes.update_padded(verts)
        #     pre_multi_view, random_view_id = get_multi_view_silhouatte(final_meshes, centroid, scale, gt_pose_params[:, 4], gt_pose_params[:, 3])
        #     loss_multi_view_silhouette = get_iou_loss(pre_multi_view, gt_multi_view[:, random_view_id, :, :])
        # else:
        loss_multi_view_silhouette = torch.tensor(0.).cuda()

        loss_chamfer_init_pcs = loss_chamfer_init_pcs * opt.chamfer_weight
        loss_chamfer = loss_chamfer * opt.chamfer_weight / num_meshes
        loss_normal = loss_normal * opt.normal_weight/ num_meshes
        loss_edge =loss_edge * opt.edge_weight/ num_meshes
        loss_smooth = loss_smooth * opt.smooth_weight/ num_meshes
        loss_multi_view_silhouette = loss_multi_view_silhouette * opt.silhouette_weight
        # loss_normal_consistency = loss_normal_consistency*opt.normal_consistency_weight/ num_meshes
        # loss_silhouette = loss_silhouette * opt.silhouette_weight / num_meshes
        # loss_depth_l1 = loss_depth_l1 * opt.depth_weight / num_meshes
        # loss_silhouette = torch.tensor(0).cuda()
        # loss_triplet = loss_triplet * opt.triplet_weight
        

        sum_loss = loss_chamfer + loss_normal + loss_edge + loss_smooth + \
                    loss_chamfer_init_pcs + loss_multi_view_silhouette
        
        
        
        train_loss_avg_log['sum'].update(sum_loss.item(), trg_im.size(0))
        train_loss_avg_log['chamfer'].update(loss_chamfer.item(), trg_im.size(0))
        train_loss_avg_log['normal'].update(loss_normal.item(), trg_im.size(0))
        train_loss_avg_log['edge'].update(loss_edge.item(), trg_im.size(0))
        train_loss_avg_log['smooth'].update(loss_smooth.item(), trg_im.size(0))
        train_loss_avg_log['init chamfer'].update(loss_chamfer_init_pcs.item(), trg_im.size(0))
        train_loss_avg_log['multi_view'].update(loss_multi_view_silhouette.item(), trg_im.size(0))

        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            print("Epoch %3d - Iter [%d/%d] Train loss: %.2f" %
                    (epoch, i + 1, train_dataloader_size, sum_loss.item()), end=' || ')
            print("Chamfer: %.4f || Noraml: %.4f || Edge: %.4f || Move: %.4f"%(loss_chamfer.item(), loss_normal.item(), loss_edge.item(), loss_smooth.item()), end=' || ')
            print("Multi_View: %.4f"%(loss_multi_view_silhouette.item()))

        if i % 200 == 0:
            # vis_img = np.array(trg_pil_list[0]).transpose([2,0,1])
            vis_img = trg_im[0].data.cpu()
            vis.image(vis_img, win='INPUT IMAGE TRAIN', opts=dict(title="INPUT IMAGE TRAIN", width=opt.win_size, height=opt.win_size))
            # if epoch>=100:
            #     vis.image(gt_multi_view[0, random_view_id[0], :, :].data.cpu(), win='silhouette gt train', opts=dict(title="silhouette gt train", width=opt.win_size, height=opt.win_size))
            #     vis.image(pre_multi_view[0, 0,:, :].data.cpu(), win='silhouette pred train', opts=dict(title="silhouette pred train", width=opt.win_size, height=opt.win_size))
            src_pc = sample_points_from_meshes(src_meshes, opt.point_num)
            pred_pc = sample_points_from_meshes(pred_meshes, opt.point_num, return_normals=False)
            vis.scatter(X=src_pc[0].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(title="TRAIN_INPUT", markersize=2, width=opt.win_size, height=opt.win_size),
                        )
            vis.scatter(X=pred_pc[0].data.cpu(),
                        win='TRAIN_OUTPUT',
                        opts=dict(title="TRAIN_OUTPUT", markersize=2, width=opt.win_size, height=opt.win_size),
                        )
            vis.scatter(X=trg_pc[0].data.cpu(),
                        win='TRAIN_GT',
                        opts=dict(title="TRAIN_GT", markersize=2, width=opt.win_size, height=opt.win_size),
                        )
            # vis.scatter(X=pred_init_pcs[0].data.cpu(),
            #             win='TRAIN_INIT_PCS',
            #             opts=dict(title="TRAIN_INIT_PCS", markersize=2, width=opt.win_size, height=opt.win_size),
            #             )
            # vis.scatter(X=negative_points[0].data.cpu(),
            #             win='TRAIN_NEGATIVE',
            #             opts=dict(title="TRAIN_NEGATIVE", markersize=2, width=opt.win_size, height=opt.win_size),
            #             )
            

    plot_data_train['X'].append(epoch)
    plot_data_train['Y'].append([train_loss_avg_log[item].avg for item in train_loss_avg_log.keys()])
    vis.line(X=np.stack([np.array(plot_data_train['X'])]*len(plot_data_train['legend']),1),
             Y=np.array(plot_data_train['Y']),
             opts={'title': opt.name + ' train loss over time',
                'legend': plot_data_train['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss',
                'width': opt.win_size, 
                'height': opt.win_size},
             win='Train loss')

    return train_loss_avg_log['sum'].avg
# ========================================================== #

def test(data_loader, trg_classes):
    val_distance = {}
    val_init_pcs_distance = {}
    val_multi_view = {}
    for cat in trg_classes:
        val_distance[cat] = AverageValueMeter()
        val_multi_view[cat] = AverageValueMeter()
        val_init_pcs_distance[cat] = AverageValueMeter()

    with torch.no_grad():
        model.eval()
        for i, data in enumerate(data_loader):
            trg_im, trg_mask, trg_pc, trg_normals, src_meshes, gt_pose_params, scale, centroid, _, trg_cat_list, gt_multi_view = data
            trg_im = trg_im.cuda()
            trg_mask = trg_mask.cuda()
            trg_pc = trg_pc.cuda()
            trg_normals = trg_normals.cuda()
            # src_meshes = src_meshes.cuda()
            gt_pose_params = gt_pose_params.cuda()
            scale = scale.cuda()
            centroid = centroid.cuda()
            gt_multi_view = gt_multi_view.cuda()
            src_meshes = ico_sphere(4, trg_im.device).extend(trg_im.shape[0])
            
            R_gt, T_gt = look_at_view_transform(dist=torch.ones(trg_im.shape[0]).cuda()*1.38, elev=gt_pose_params[:, 1], azim=gt_pose_params[:, 0])
            cameras_gt = FoVPerspectiveCameras(device=device, R=R_gt, T=torch.zeros_like(T_gt).cuda(), fov=gt_pose_params[:, 3])
            transform_gt = cameras_gt.get_world_to_view_transform()
            transform_gt_inverse = transform_gt.inverse()
            rotate_centroid = transform_gt.transform_points(centroid.unsqueeze(1)).detach()
            rotate_centroid = rotate_centroid.squeeze()
            cameras = FoVPerspectiveCameras(device=device, R=torch.eye(3).unsqueeze(0).expand(trg_im.shape[0], 3, 3), T=T_gt, fov=gt_pose_params[:, 3])
            pred_meshes_list, pred_init_pcs, _ = model(trg_im, src_meshes, cameras, scale, rotate_centroid)
            # pred_meshes_list, _ = model(trg_im, src_meshes, cameras, scale, rotate_centroid)
            pred_meshes = pred_meshes_list[-1]
            verts = pred_meshes.verts_padded()
            if not torch.isfinite(verts).all():
                print("!!!!!!!!!!!!!!!!!!!!!!!!!Test Found NaN!!!!!!!!!!!!!!!!!!!!!!!!!")
                for stage in range(1,3):
                    for m in model.sphere_deform_layers[stage].modules():
                        if isinstance(m, torch.nn.BatchNorm1d):
                            print(m)
                            torch.nn.init.constant_(m.running_var, 1)
                            torch.nn.init.constant_(m.running_mean, 0)
                continue

            pred_verts_canonical = transform_gt_inverse.transform_points(verts)
            # _centroid = torch.mean(pred_verts_canonical, dim=1)
            # pred_verts_canonical = pred_verts_canonical - _centroid.unsqueeze(1).expand_as(pred_verts_canonical)
            # _scale = torch.max(torch.sqrt(torch.sum(pred_verts_canonical ** 2, dim=2)), dim=1)[0] + 1e-4
            # pred_verts_canonical = pred_verts_canonical / _scale.unsqueeze(1).unsqueeze(1)
            pred_meshes = pred_meshes.update_padded(pred_verts_canonical)

            pred_pc = sample_points_from_meshes(pred_meshes, opt.point_num)
            chamfer_dist1, chamfer_dist2, _, _= distChamfer(trg_pc, pred_pc)
            val_loss = torch.mean(chamfer_dist1 + chamfer_dist2, 1)

            # pred_init_pcs = transform_gt_inverse.transform_points(pred_init_pcs)
            # pred_init_pcs = pred_init_pcs*torch.FloatTensor([[-1,1,-1]]).cuda().unsqueeze(0)
            # chamfer_dist1, chamfer_dist2, _, _= distChamfer(trg_pc, pred_init_pcs)
            # val_pcs_chamfer = torch.mean(chamfer_dist1 + chamfer_dist2, 1)

            # pred_pose = get_pred_angle(pose_params)
            # pred_silhouette = get_render_img(azim=pred_pose[:, 0], elev=pred_pose[:, 1], distance=gt_pose_params[:, 2], view=gt_pose_params[:, 3], 
            #                                     mesh=pred_meshes, multi_view_distance=gt_pose_params[:, 4], raw_scale=scale, raw_centroid=centroid, render_rgb=False, train=False)

            # pred_verts_canonical = transform_gt_inverse.transform_points(pred_meshes.verts_padded())
            # pred_verts_canonical = pred_verts_canonical*torch.FloatTensor([[-1,1,-1]]).cuda().unsqueeze(0)
            # pred_meshes = pred_meshes.update_padded(pred_verts_canonical)
            pred_multi_view_silhouette = get_multi_view_silhouatte_test(distance=gt_pose_params[:, 4], view=gt_pose_params[:, 3], mesh=pred_meshes, raw_scale=scale, raw_centroid=centroid)
            multi_view_iou = get_iou(pred_multi_view_silhouette, gt_multi_view, size_average=False)

            # print(val_loss)
            for batch_i in range(len(trg_cat_list)):
                val_distance[trg_cat_list[batch_i]].update(val_loss[batch_i].item())
                # val_init_pcs_distance[trg_cat_list[batch_i]].update(val_pcs_chamfer[batch_i].item())
                val_multi_view[trg_cat_list[batch_i]].update(multi_view_iou[batch_i].item())

            if (i + 1) % 10 == 0:
                print("Epoch %3d - Iter [%d/%d] Val dist: %.4f" % (epoch, i + 1, val_dataloader_size, torch.mean(val_loss).item()))
                # print("Epoch %3d - Iter [%d/%d] Val dist: %.4f" % (epoch, i + 1, val_dataloader_size, torch.mean(val_loss).item()))

            if i % 100 == 0:
                # vis_img = np.array(trg_pil_list[0]).transpose([2,0,1])
                vis_img = trg_im[0].data.cpu()
                vis.image(vis_img, win='INPUT IMAGE VAL', opts=dict(title="INPUT IMAGE VAL", width=opt.win_size, height=opt.win_size))
                # vis.image(pred_rgb[0].data.cpu().numpy().transpose([2,0,1]), win='RGB VAL', opts=dict(title="RGB VAL", width=opt.win_size, height=opt.win_size))
                # vis.image(pred_silhouette[0].data.cpu(), win='silhouette val', opts=dict(title="silhouette val", width=opt.win_size, height=opt.win_size))
                src_pc = sample_points_from_meshes(src_meshes, opt.point_num)
                input_vis_x = src_pc[0].data.cpu()
                pred_vis_x = pred_pc[0].data.cpu()
                gt_vis_x = trg_pc[0].data.cpu()
                vis.scatter(X=input_vis_x,
                            win='VAL_INPUT',
                            opts=dict(title="VAL_INPUT", markersize=2, width=opt.win_size, height=opt.win_size),
                            )
                vis.scatter(X=pred_vis_x,
                            win='VAL_OUTPUT',
                            opts=dict(title="VAL_OUTPUT", markersize=2, width=opt.win_size, height=opt.win_size),
                            )
                # vis.scatter(X=pred_init_pcs[0].data.cpu(),
                #             win='VAL_INIT_PCS',
                #             opts=dict(title="VAL_INIT_PCS", markersize=2, width=opt.win_size, height=opt.win_size),
                #             )
                vis.scatter(X=gt_vis_x,
                            win='VAL_GT',
                            opts=dict(title="VAL_GT", markersize=2, width=opt.win_size, height=opt.win_size),
                            )
    mean_chamfer = []
    # init_mean_chamfer = []
    mean_iou = []
    for cat in trg_classes:
        mean_chamfer.append(val_distance[cat].avg)
        # init_mean_chamfer.append(val_init_pcs_distance[cat].avg)
        mean_iou.append(val_multi_view[cat].avg)
        print(cat, val_distance[cat].avg, val_init_pcs_distance[cat].avg, val_multi_view[cat].avg)

    plot_data_val['X'].append(epoch)
    plot_data_val['Y'].append([val_distance[cat].avg for cat in trg_classes]+[np.mean(mean_chamfer)])
    # plot_data_val['Init PCs'].append([val_init_pcs_distance[cat].avg for cat in trg_classes]+[np.mean(init_mean_chamfer)])
    # plot_data_val['Pose'].append([val_pose_similarity[cat].avg for cat in trg_classes]+[np.mean(pose_mean)])
    plot_data_val['IoU'].append([val_multi_view[cat].avg for cat in trg_classes]+[np.mean(mean_iou)])
    vis.line(X=np.stack([np.array(plot_data_val['X'])]*len(plot_data_val['legend']),1),
             Y=np.array(plot_data_val['Y']),
             opts={'title': opt.name + ' val loss over time',
                'legend': plot_data_val['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss',
                'width': opt.win_size, 
                'height': opt.win_size},
             win='Validation loss')
    # vis.line(X=np.stack([np.array(plot_data_val['X'])]*len(plot_data_val['legend']),1),
    #          Y=np.array(plot_data_val['Init PCs']),
    #          opts={'title': opt.name + ' val init pcs chamfer',
    #             'legend': plot_data_val['legend'],
    #             'xlabel': 'epoch',
    #             'ylabel': 'loss',
    #             'width': opt.win_size, 
    #             'height': opt.win_size},
    #          win='init pcs')
    vis.line(X=np.stack([np.array(plot_data_val['X'])]*len(plot_data_val['legend']),1),
             Y=np.array(plot_data_val['IoU']),
             opts={'title': opt.name + ' multi-view IoU',
                'legend': plot_data_val['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss',
                'width': opt.win_size, 
                'height': opt.win_size},
             win='iou')
    
    return np.mean(mean_chamfer)

# =============BEGIN OF THE LEARNING LOOP=================== #
# initialization

best_val_dist = 0.0068
optimizer = torch.optim.Adam(
    params=list(model.parameters()),
    lr=opt.lr,
    betas=(0.9, 0.999),
    weight_decay=1.0E-6
)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [100, 200, 230], 0.3
)

loed_model_name = '/home/xianghui_yang/viewer_centered/log/2022-03-08_0shot_viewer_no_distance/model-99-0.0080.pth'
pretrained_dict = torch.load(loed_model_name)
model.load_state_dict(pretrained_dict, strict=True)
print("Previous weight loaded: ", loed_model_name)
loed_model_name = '/home/xianghui_yang/viewer_centered/log/2022-03-08_0shot_viewer_no_distance/optimizer-99-0.0080.pth'
pretrained_dict = torch.load(loed_model_name)
optimizer.load_state_dict(pretrained_dict)
print("Previous weight loaded: ", loed_model_name)
loed_model_name = '/home/xianghui_yang/viewer_centered/log/2022-03-08_0shot_viewer_no_distance/lr_scheduler-99-0.0080.pth'
pretrained_dict = torch.load(loed_model_name)
lr_scheduler.load_state_dict(pretrained_dict)
print("Previous weight loaded: ", loed_model_name)
    
for epoch in range(opt.n_epoch):
    # train
    start = time.time()
    train_loss_avg = train(train_loader, optimizer)
    val_dist = test(val_loader, opt.novel_class)
    print("Epoch %3d - Val Mean dist: %.4f" % (epoch, val_dist))
    lr_scheduler.step()
    print('Current Lr: ', lr_scheduler.get_last_lr()[0])
    
    if val_dist<=best_val_dist:
        torch.save(model.state_dict(), os.path.join(result_path, 'model-%03d-%.4f.pth'%(epoch, val_dist)))
        torch.save(optimizer.state_dict(), os.path.join(result_path, 'optimizer-%03d-%.4f.pth'%(epoch, val_dist)))
        torch.save(lr_scheduler.state_dict(), os.path.join(result_path, 'lr_scheduler-%03d-%.4f.pth'%(epoch, val_dist)))
        print("Better Model Saved")
        best_val_dist = val_dist
    
    print('Epoch: %03d|| Used Time:%.2f mins\n' % (epoch, (time.time()-start)/60))
    if epoch==99:
        torch.save(model.state_dict(), os.path.join(result_path, 'model-99-%.4f.pth'%(val_dist)))
        torch.save(optimizer.state_dict(), os.path.join(result_path, 'optimizer-99-%.4f.pth'%(val_dist)))
        torch.save(lr_scheduler.state_dict(), os.path.join(result_path, 'lr_scheduler-99-%.4f.pth'%(val_dist)))
        print('No Multi-view IoU Saved.')

    torch.save(model.state_dict(), os.path.join(result_path, 'final-model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(result_path, 'final-optimizer.pth'))
    torch.save(lr_scheduler.state_dict(), os.path.join(result_path, 'final-lr_scheduler.pth'))