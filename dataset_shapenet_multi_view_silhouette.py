import os
import scipy.io as sio
import numpy as np
from PIL import Image
import random
random.seed(2021)
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import json5
from pytorch3d.structures import join_meshes_as_batch
import glob

HEIGHT_PATCH = 224
WIDTH_PATCH = 224

data_transforms_nocrop = transforms.Compose([
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.ToTensor()
])
data_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class ShapeNet(data.Dataset):
    def __init__(self, input_size=224, train=True, target_class=None, point_num=2500):
        # self.train = train
        # self.root_dir = root_dir
        self.input_size = input_size
        self.point_num = point_num
        self.cat2idx = {}
        # self.template_path = {}
        self.template_fn = {}
        self.template_meshes = {}
        self.train = train
        self.meta = []
        self.trg_src = {}

        with open('/home/xianghui_yang/data/id2cat.json', 'r') as file:
            self.idx2cat = json.load(file)
        print(self.idx2cat)
        for idx in self.idx2cat.keys():
            self.cat2idx[self.idx2cat[idx]]=idx

        if not self.train:
            # with open('/home/xianghui_yang/data/val_trg_src_pairs.json', 'r') as file:
            #     trg_src_dict = json.load(file) # cat: [[trg_id, src_id], [trg_id, src_id], ...]
            # for cat in trg_src_dict:
            #     self.trg_src[cat]={}
            #     for pair in trg_src_dict[cat]:
            #         self.trg_src[cat][pair[0]]=pair[1]
            with open('/home/xianghui_yang/data/train_val_minmum_trg_src_pairs.json', 'r') as file:
                self.trg_src = json.load(file)

        for cat in target_class:
            cat_index = self.cat2idx[cat]
            dir_img  = '/home/xianghui_yang/data/ShapeNet/ShapeNetRendering/%s'%cat_index
            dir_multi_view_mask = '/home/xianghui_yang/data/ShapeNet/Pytorch3D_Silhouette_Render/%s'%cat_index
            dir_mat = '/home/xianghui_yang/data/ShapeNet/ShapeNetV1PointCloud_Normalized/%s'%cat_index
            fns_img = sorted(os.listdir(dir_img))
            fns_mat = sorted(os.listdir(dir_mat))
            fns = [val for val in fns_img if val+'.mat' in fns_mat]
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[-200:]
            print(cat, 'valid num: ', len(fns), len(fns)/len(fns_mat))
            
            for fn in fns:
                # self.meta.append((os.path.join(dir_img, fn, "rendering"), os.path.join(dir_obj, fn, 'models', 'model_normalized.obj'), fn))
                # self.meta.append((os.path.join(dir_img, fn, "rendering"), os.path.join(dir_mat, fn+'.mat'), fn, cat))
                tmp_mat = sio.loadmat(os.path.join(dir_mat, fn+'.mat'))
                # if tmp_mat['verts'].shape[0]<5e4 and tmp_mat['faces'].shape[0]<15e4:
                #     self.meta.append((os.path.join(dir_img, fn, "rendering"), os.path.join(dir_mat, fn+'.mat'), fn, cat))
                pc = torch.from_numpy(tmp_mat['pc'])*torch.FloatTensor([[1,1,-1]])
                normal = torch.from_numpy(tmp_mat['n'])*torch.FloatTensor([[1,1,-1]])
                scale = torch.from_numpy(tmp_mat['scale'][0])
                centroid = torch.from_numpy(tmp_mat['centroid'][0])*torch.FloatTensor([1,1,-1])
                self.meta.append((os.path.join(dir_img, fn, "rendering"), pc, normal, fn, cat, scale, centroid, os.path.join(dir_multi_view_mask, fn)))
                    
            print(cat, 'valid mesh sum: ', len(self.meta))

        self.data_num = len(self.meta)
        self.im_transform = data_transforms_nocrop
        self.im_normalize = data_normalize
        # self.render_transform = transforms.ToTensor()
        # if input_size != 224:
        #     self.render_transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        trg_infor = self.meta[idx]
        img_path = trg_infor[0]
        trg_points = trg_infor[1]
        trg_normals = trg_infor[2]
        fn = trg_infor[3]
        cat = trg_infor[4]
        scale = trg_infor[5]
        centroid = trg_infor[6]
        multi_view_path = trg_infor[7]

        # trg_mat = sio.loadmat(trg_infor[1])
        # trg_points = torch.from_numpy(trg_mat['pc'])*torch.FloatTensor([[1,1,-1]])
        # trg_normals = torch.from_numpy(trg_mat['n'])*torch.FloatTensor([[1,1,-1]])
        indices = np.random.randint(trg_points.shape[0], size=self.point_num)
        trg_points = trg_points[indices,:]
        trg_normals = trg_normals[indices,:]
        

        files = glob.glob(os.path.join(img_path, '*.png'))
        files = sorted(files)
        if self.train:
            idx_img = np.random.randint(0, len(files)-1)
        else:
            idx_img = 0
        filename = files[idx_img]
        trg_im = self.im_transform(Image.open(filename))
        # ssim_filename = os.path.join('/home/xianghui_yang/data/ShapeNet/Pytorch3D_Render', self.cat2idx[cat], fn, '%02d.jpg'%idx_img)
        # ssim_im = self.im_transform(Image.open(ssim_filename))
        with open(os.path.join(img_path, 'rendering_metadata.txt'),'r') as f:
            param_list = f.readlines()
            multi_view_dist = float(param_list[0].rstrip().split(' ')[3])*1.75
            param = param_list[idx_img].rstrip() 
            param = param.split(' ')
            
        azim = float(param[0]) #+ random.gauss(0, 2)
        elev = float(param[1]) #+ random.gauss(0, 0.5)
        distance = float(param[3])*1.75
        view = float(param[4])*2
        cam_params = [azim, elev, distance, view, multi_view_dist]
        trg_mask = trg_im[3, :, :]
        trg_mask[trg_mask>0]=1

        multi_view_files_list = glob.glob(os.path.join(multi_view_path, '*.jpg'))
        multi_view_files_list = sorted(multi_view_files_list)
        multi_view_silhouette = []
        for multi_view_file in multi_view_files_list:
            multi_view_file = self.im_transform(Image.open(multi_view_file))
            multi_view_silhouette.append(multi_view_file)
        multi_view_silhouette = torch.cat(multi_view_silhouette, dim=0)

        return trg_im[:3, :, :], trg_mask, trg_points, trg_normals, src_mesh, cam_params, scale, centroid, fn, cat, multi_view_silhouette



def collate_fn(batch):
    trg_im = torch.stack([elem[0] for elem in batch])
    # trg_ssim_im = torch.stack([elem[1] for elem in batch])
    trg_mask = torch.stack([elem[1] for elem in batch])
    trg_points = torch.stack([elem[2] for elem in batch])
    trg_normals = torch.stack([elem[3] for elem in batch])
    # pil_img_list= [elem[3] for elem in batch]

    # src_verts_list = [elem[4] for elem in batch]
    # src_faces_idx_list = [elem[5] for elem in batch]
    # src_meshes = Meshes(verts=src_verts_list, faces=src_faces_idx_list)
    src_meshes = join_meshes_as_batch([elem[4] for elem in batch])
    cam_params = torch.FloatTensor([elem[5] for elem in batch])
    scale = torch.stack([elem[6] for elem in batch])
    centroid = torch.stack([elem[7] for elem in batch])
    fn_list = [elem[8] for elem in batch]
    cat_list = [elem[9] for elem in batch]
    multi_view_silhouette = torch.stack([elem[10] for elem in batch])
    
    return trg_im, trg_mask, trg_points, trg_normals, src_meshes, cam_params, scale, centroid, fn_list, cat_list, multi_view_silhouette

    # query_im = torch.stack([elem[0] for elem in batch])
    # trg_pc = torch.stack([elem[1] for elem in batch])
    # trg_rgb= [elem[2] for elem in batch]
    # src_pc = torch.stack([elem[3] for elem in batch])
    # return query_im, trg_pc, trg_rgb, src_pc
