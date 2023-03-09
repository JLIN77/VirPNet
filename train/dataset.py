import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from config import config

GRID_SIZE = config.GRID_SIZE
MAX_DISTANCE = config.MAX_DISTANCE
MAX_DISTANCE_X = config.MAX_DISTANCE_X
MAX_DISTANCE_Y = config.MAX_DISTANCE_Y
MAX_DISTANCE_Z = config.MAX_DISTANCE_Z

# instance_path = './instance_npz'
# label_path = './instance_label'
# instance_train = './instance_dataset/train'
# instance_val = './instance_dataset/val'


def get_im_min_shape():
    instance_path = './instance_npz'
    instance_ims_name =[]
    for npz_name in os.listdir(instance_path):
        instance_im_name = npz_name[:-4] + '.png'
        instance_ims_name.append(instance_im_name)
    # print(instance_ims_name)  # 1776 instance

    kitti_path = './dataset/KITTI'
    image_path = kitti_path + '/object/training/image_2'
    ims_shape = []
    for im_name in tqdm(instance_ims_name):
        im_name_full = os.path.join(image_path, im_name)
        im = cv2.imread(im_name_full)
        h, w, c = im.shape
        ims_shape.append([w, h])
    # print(ims_shape)
    ims_shape = np.array((ims_shape))
    # print(ims_shape.shape)

    im_min_shape = (min(ims_shape[:, 0]), min(ims_shape[:, 1]))
    # print(im_min_shape)
    return im_min_shape


# im_min_shape = get_im_min_shape()  # [1224, 370]


def read_info_from_data_npz(npz_name_full):
    # npz_name_full = os.path.join(instance_npz, npz_name)
    info = np.load(npz_name_full)
    im_data = info['data']
    im_name = info['im_name']
    im_width = info['im_width']
    im_height = info['img_height']

    return im_data, str(im_name), im_width, im_height


def read_info_from_label_npz(npz_name_full):
    info = np.load(npz_name_full)

    virtual_lidar_map = info['virtual_lidar_map']
    im_name = info['im_name']

    # w, h = real_lidar_map.shape[0], real_lidar_map.shape[1]
    # # exit()
    # for u in range(w):
    #     for v in range(h):
    #         pc = real_lidar_map[u, v]
    #         if pc[0] != 0 and pc[1] != 0 and pc[0] != 0:
    #             print(pc)
    return virtual_lidar_map, str(im_name)


transform = transforms.Compose([transforms.ToTensor()])


class LinDataset(Dataset):
    def __init__(self, data_path, label_path, test=False, min_shape=(1224, 370), transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.test = test
        self.min_shape = min_shape
        self.transform = transform
        self.npz_name_list = self.get_npz_name_list()

    def get_npz_name_list(self):
        return os.listdir(self.data_path)

    def __getitem__(self, index):
        npz_name = self.npz_name_list[index]
        # print(npz_name)

        # load data
        npz_name_full_data = os.path.join(self.data_path, npz_name)
        im_data, im_name, im_width, im_height = read_info_from_data_npz(npz_name_full_data)

        im_data = im_data.transpose([2, 0, 1])  # w, h, c -> c, w, h
        # f_map, b_map = im_data[4, :, :], im_data[5, :, :]
        uv_map = im_data[18, :, :]

        # load label
        npz_name = npz_name[:-4] + '_label.npz'
        npz_name_full_label = os.path.join(self.label_path, npz_name)
        # label_map, virtual_lidar_map, real_lidar_map, im_name = read_info_from_label_npz(npz_name_full_label)
        virtual_lidar_map, im_name = read_info_from_label_npz(npz_name_full_label)
        # print(label_map.shape)

        # cut data and label into min_shape (top-left delete if necessery)
        cut_width = im_width - self.min_shape[0]
        cut_height = im_height - self.min_shape[1]
        cut_data = im_data[:, cut_width:, cut_height:]
        cut_uv_map = uv_map[cut_width:, cut_height:]
        # print('--', np.sum(cut_uv_map))

        cut_virtual_lidar_map = virtual_lidar_map[cut_width:, cut_height:, :]

        # # data normalization
        cut_data[0:3, :, :] = cut_data[0:3, :, :] / 255.0       # RGB     3
        cut_data[3, :, :] = cut_data[3, :, :]                   # k_map   1 + 4
        cut_data[4, :, :] /= MAX_DISTANCE_X
        cut_data[5, :, :] /= MAX_DISTANCE_Y
        cut_data[6, :, :] /= MAX_DISTANCE_Z
        cut_data[7, :, :] = cut_data[7, :, :]
        cut_data[8, :, :] = cut_data[8, :, :]                   # f_map   1 + 4
        cut_data[9, :, :] /= MAX_DISTANCE_X
        cut_data[10, :, :] /= MAX_DISTANCE_Y
        cut_data[11, :, :] /= MAX_DISTANCE_Z
        cut_data[12, :, :] = cut_data[12, :, :]
        cut_data[13, :, :] = cut_data[13, :, :]                 # b_map   1 + 4
        cut_data[14, :, :] /= MAX_DISTANCE_X
        cut_data[15, :, :] /= MAX_DISTANCE_Y
        cut_data[16, :, :] /= MAX_DISTANCE_Z
        cut_data[17, :, :] = cut_data[17, :, :]
        cut_data[18, :, :] = cut_data[18, :, :]                 # uv_map 1 + 2
        cut_data[19:21, :, :] = cut_data[19:21, :, :] / GRID_SIZE

        # for i in range(18):
        #     save = pd.DataFrame(cut_data[i, :, :].T)
        #     save_name = './cut_data/cut_data_' + str(i) + '.csv'
        #     save.to_csv(save_name)
        # exit()

        if self.transform:
            cut_data = transform(cut_data)

        if self.test:
            k_pt_map = cut_data[7, :, :]
            return cut_data, cut_uv_map, cut_virtual_lidar_map, im_name

    def __len__(self):
        return len(self.npz_name_list)


# if __name__ == '__main__':
#     dataset_train = LinDataset(data_path=instance_val, label_path=label_path, test=True)
#     dataloader_train = DataLoader(dataset_train, shuffle=False, batch_size=1, num_workers=1)
#
#     for i, (data, label) in tqdm(enumerate(dataloader_train)):
#         print(data.shape, label.shape)
#         if i == 10:
#             exit()
