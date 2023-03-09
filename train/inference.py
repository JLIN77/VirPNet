from tqdm import tqdm
import numpy as np

import torch
import time
from torch.utils.data import DataLoader
from net import Linnet
from dataset import LinDataset

from config import config

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(ROOT_DIR)


MAX_DISTANCE = config.MAX_DISTANCE


def filter_fuc(pred_dl_map, virtual_lidar_map):
    # filter the background lidar points
    pred_dl_map_mask = np.where(np.abs(pred_dl_map) > 0.2, 0, 1)
    pred_dl_map = pred_dl_map * pred_dl_map_mask

    for i in range(3):
        virtual_lidar_map[:, :, i] *= pred_dl_map_mask

    return pred_dl_map, virtual_lidar_map


def get_predict_pc(dl, virtual_pc):
    x0, y0, z0 = virtual_pc
    d0 = np.sqrt(x0**2+y0**2+z0**2)
    k = dl/d0
    x, y, z = x0+x0*k, y0+y0*k, z0+z0*k

    # print('-', x0, y0, z0)
    # print('+', dl, d0, k)
    # print('-', x, y, z)
    # exit()
    return [x, y, z0]


def show_results(datas, pred_dl_maps, im_names, virtual_lidar_maps, real_lidar_maps):
    for i, pred_dl_map in enumerate(pred_dl_maps):
        im_name, virtual_lidar_map, real_lidar_map = im_names[i], virtual_lidar_maps[i], real_lidar_maps[i]
        pred_dl_map = pred_dl_map.detach().cpu().numpy()
        virtual_lidar_map = virtual_lidar_map.detach().cpu().numpy()
        real_lidar_map = real_lidar_map.detach().cpu().numpy()

        fb_map = datas[i, 8, :, :]
        fb_map = fb_map.detach().cpu().numpy()

        print(virtual_lidar_map.shape)

        # real
        pcd_real = []
        real_map_idx = np.nonzero(real_lidar_map)
        real_map_idx = np.array(real_map_idx).transpose([1, 0])
        for idx in real_map_idx:
            pc_real = real_lidar_map[idx[0], idx[1], :]
            pcd_real.append(pc_real)
        pcd_real = np.array(pcd_real)

        # virtual
        pcd_virtual = []
        virtual_map_idx = np.nonzero(virtual_lidar_map)
        virtual_map_idx = np.array(virtual_map_idx).transpose([1, 0])
        for idx in virtual_map_idx:
            pc_virtual = virtual_lidar_map[idx[0], idx[1], :]
            pcd_virtual.append(pc_virtual)
        pcd_virtual = np.array(pcd_virtual)

        # pred
        pred_dl_map, virtual_lidar_map = filter_fuc(pred_dl_map, virtual_lidar_map)

        map0 = virtual_lidar_map[:, :, 0]
        map1 = virtual_lidar_map[:, :, 1]
        map2 = virtual_lidar_map[:, :, 2]
        map0 = np.where(map0 != 0, 1, 0)
        map1 = np.where(map1 != 0, 1, 0)
        map2 = np.where(map2 != 0, 1, 0)
        map = map0 | map1 | map2

        map_idx = np.nonzero(map)
        map_idx = np.array(map_idx).transpose([1, 0])

        pcd_predict = []
        for idx in map_idx:
            pc_virtual = virtual_lidar_map[idx[0], idx[1], :]
            dl = pred_dl_map[idx[0], idx[1]]
            pc_pred = get_predict_pc(dl, pc_virtual)
            pcd_predict.append(pc_pred)
            # print(dl)
        pcd_predict = np.array(pcd_predict)

        # save npy
        idx = str(im_name)[:-4]
        save_path = config.inference_res + '/' + idx
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        print(pcd_real.shape)
        print(pcd_virtual.shape)
        print(pcd_predict.shape)
        # print(pcd_predict)

        pcd_real = np.array(pcd_real)
        np.save(save_path + '/{}_real.npy'.format(im_name[:-4]), pcd_real)
        pcd_predict = np.array(pcd_predict)
        np.save(save_path + '/{}_pred.npy'.format(im_name[:-4]), pcd_predict)
        pcd_virtual = np.array(pcd_virtual)
        np.save(save_path + '/{}_virtual.npy'.format(im_name[:-4]), pcd_virtual)

        data = datas[i].detach().cpu().numpy()
        data = data * 255
        k_map = data[3, :, :]
        k_map_idx = np.nonzero(k_map)
        k_map_idx = np.array(k_map_idx).transpose([1, 0])

        pcd_kpoint = []
        for k in k_map_idx:
            # print(k)
            pc_kpoint = real_lidar_map[k[0], k[1], :]
            pcd_kpoint.append(pc_kpoint)
        # print(pcd_kpoint)
        pcd_kpoint= np.array(pcd_kpoint)
        np.save(save_path + '/{}_kpoint.npy'.format(im_name[:-4]), pcd_kpoint)

        # exit()

        # pcd = pcd_real
        # show_lidar(pcd)


def get_pc_cls(u, v, cls_map, THRESHOLD=0.5):
    # print(cls_map[0, u, v], cls_map[1, u, v])
    if cls_map[0, u, v] > THRESHOLD:
        return 'f'
    elif cls_map[1, u, v] > THRESHOLD:
        return 'b'
    else:
        return 'fb'


def decode_dl(dl):
    dl = np.log((1+dl)/(1-dl))
    return dl


def read_kfb_points_from_instance_pts_npz(instance_pts_npz):
    instance_pts_npz_name_full = os.path.join(config.instance_pts_npz, instance_pts_npz)
    info = np.load(instance_pts_npz_name_full, allow_pickle=True)

    k_pts = info['k_points']
    f_pts = info['f_points']
    b_pts = info['b_points']
    r_pts = info['r_points']

    # print(k_pts.shape, f_pts.shape, b_pts.shape)
    k_pts_3d, f_pts_3d, b_pts_3d = [], [], []
    if k_pts.shape[0] != 0:
        k_pts_3d = k_pts[:, 1]
    if f_pts.shape[0] != 0:
        f_pts_3d = f_pts[:, 1]
    if b_pts.shape[0] != 0:
        b_pts_3d = b_pts[:, 1]

    k_pcd, f_pcd, b_pcd = [], [], []
    for pc in k_pts_3d:
        k_pcd.append(list(pc))
    for pc in f_pts_3d:
        f_pcd.append(list(pc))
    for pc in b_pts_3d:
        b_pcd.append(list(pc))

    return k_pcd, f_pcd, b_pcd


def decode_results(cls_maps, reg_maps, uv_maps, im_names, virtual_lidar_maps, f_score=0.3, dl_threshold=0.5):
    batch_size = cls_maps.shape[0]
    for i in range(batch_size):
        cls_map = cls_maps[i].detach().cpu().numpy()                       # torch.Size([2, 1224, 370])
        reg_map = reg_maps[i].detach().cpu().numpy()
        uv_map = uv_maps[i]
        im_name = im_names[i]
        virtual_lidar_map = virtual_lidar_maps[i].detach().cpu().numpy()   # torch.Size([1224, 370, 3])

        # print(virtual_lidar_map.shape, cls_map.shape)

        f_map = np.where(cls_map[0] > f_score, 1, 0)
        print(f_map.shape)

        # exit()

        # uv_map_idx = np.nonzero(uv_map)
        # uv_map_idx = np.array(uv_map_idx)

        f_map_idx = np.nonzero(f_map)
        # print(f_map_idx)
        f_map_idx_list = []
        for i in range(f_map_idx[0].shape[0]):
            f_map_idx_list.append([f_map_idx[0][i], f_map_idx[1][i]])

        pcd_real, pcd_virt, pcd_pred, pcd_pred_4d = [], [], [], []
        num = 0
        for u, v in f_map_idx_list:
            # print(u, v)
            # exit()

            pc_virt = virtual_lidar_map[u, v, :3]
            pc_intensive = virtual_lidar_map[u, v, 3]

            pc_cls = get_pc_cls(u, v, cls_map)
            pc_dl = reg_map[u, v]
            # print(pc_dl)
            pc_dl = decode_dl(dl=pc_dl)
            # print(pc_dl)

            pcd_virt.append(pc_virt)

            if pc_cls == 'f':
                pc_pred = get_predict_pc(dl=pc_dl, virtual_pc=pc_virt)
                # print(pc_dl, 'f')
                print('pc_pred, pc_virt:\n {}\n {}\n'.format(pc_pred, pc_virt))
                # exit()
                if pc_dl < dl_threshold and pc_dl > - dl_threshold:   # filter the big dl
                    pc_pred = get_predict_pc(dl=pc_dl, virtual_pc=pc_virt)
                    # print(pc_dl, 'f')
                    # print(pc_pred)
                    # exit()
                    pcd_pred.append(pc_pred)
                    pcd_pred_4d.append(pc_pred + [pc_intensive])
                    num += 1
            elif pc_cls == 'b':
                pass
                # pc_pred = get_predict_pc(dl=pc_dl, virtual_pc=pc_virt)
                # pcd_pred.append(pc_pred)
                # pcd_pred_4d.append(pc_pred + [pc_ref])
            else:
                # print('---Neither f Nor b------')
                pass

        # if num < 50: continue
        pcd_pred_np = np.float32(pcd_pred_4d)
        pcd_real_np = np.array(pcd_real)
        pcd_virt_np = np.array(pcd_virt)
        # print(pcd_pred_np)
        # exit()

        idx = str(im_name)[:-4]
        save_path = config.inference_res + '/' + idx
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(save_path + '/{}_pred.npy'.format(im_name[:-4]), pcd_pred_np)
        np.save(save_path + '/{}_virtual.npy'.format(im_name[:-4]), pcd_virt_np)

        ###############################################
        # add kfb points
        # 1. load kfb points
        instance_pts_npz = im_name[:-4] + '_kfb_pts.npz'
        k_pcd, f_pcd, b_pcd = read_kfb_points_from_instance_pts_npz(instance_pts_npz=instance_pts_npz)
        # print(k_pcd)
        # exit()

        # k + f
        kf_pcd_np = np.array(k_pcd + f_pcd)
        np.save(save_path + '/{}_kf.npy'.format(im_name[:-4]), kf_pcd_np)
        # k + f + pred
        kfp_pcd_np = np.array(k_pcd + f_pcd + pcd_pred_4d)
        np.save(save_path + '/{}_kfp.npy'.format(im_name[:-4]), kfp_pcd_np)

        print('k_pts, f_pts, b_pts: {}, {}, {}'.format(len(k_pcd), len(f_pcd), len(b_pcd)))
        print('pcd_virt, pcd_pred: {}, {}'.format(len(pcd_virt), len(pcd_pred)))
        print('---------------------------------------------------------------')
        print('before complement, after complement: {}, {}'.format(kf_pcd_np.shape[0], kfp_pcd_np.shape[0]))


if __name__ == '__main__':

    # initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and checkpoint
    # model = Linnet()
    checkpoint_weight = config.checkpoint
    model = torch.load(checkpoint_weight)
    # model.load_state_dict(torch.load(checkpoint_weight))

    # load data
    dataset_val = LinDataset(data_path=config.instance_npz, label_path=config.label_path, test=True)
    dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=config.batch_size_test, num_workers=1)

    # model inference
    model.to(device)
    model.eval()
    for i, (datas, uv_map, virtual_lidar_maps, im_names) in tqdm(enumerate(dataloader_val)):
        inputs = datas.to(device).float()
        outputs = model(inputs)
        # predictions = outputs.squeeze(axis=1)   # [N, 1, W, H] -> [N, W, H]
        # print(predictions.shape)
        cls_map = outputs[:, 0:2, :, :]     # (B, 2, W, H),
        reg_map = outputs[:, 2, :, :]       # (B, W, H)
        # ref_map = outputs[:, 3, :, :]       # (B, W, H)
        # reg_map = reg_map.squeeze(axis=1)

        decode_results(cls_maps=cls_map, reg_maps=reg_map, uv_maps=uv_map, im_names=im_names,
                       virtual_lidar_maps=virtual_lidar_maps)
        if i == 2:
            exit()

    pass