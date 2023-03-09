"""
Annotation: this file creates instance labels from instance_npz. Instance labels are saved into instance_label folder
            as follow:
            np.savez(npz_name_full, dl_map=dl_map, virtual_lidar_map=virtual_lidar_map,
                     real_lidar_map=real_lidar_map, im_name=im_name)
Anthor: Lin, PRML Lab
Data: 20, July, 2022
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
from kitti.kitti_object import kitti_object
from utils.util import read_info_from_npz

from train.config import config

GRID_SIZE = config.GRID_SIZE
MAX_DISTANCE = config.MAX_DISTANCE
instance_npz = './instance_npz'
instance_pts_npz = './instance_pts_npz'
instance_label = './instance_label'
kitti_path = './dataset_split/KITTI'
label_path = kitti_path + '/object/training/label_2'


def read_info_form_kbf_pts_npz(npz_pts_name_full):
    info = np.load(npz_pts_name_full, allow_pickle=True)
    # print(info.files)    # ['k_points', 'f_points', 'b_points']
    k_pts = info['k_points']  # [[list([863, 230]) array([14.736, -5.15 , -1.141], dtype=float32)], ...]
    f_pts = info['f_points']
    b_pts = info['b_points']
    r_pts = info['r_points']
    #
    # print(r_pts)
    # exit()

    return k_pts, f_pts, b_pts, r_pts


def get_distance(p):
    return np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)


def sigmoid_(x):    # 2sigmoid - 1
    return 2. / (1. + np.exp(-x)) - 1


def get_virtual_lidar_map(k_pts, r_pts, im_name, w, h, grid_size=GRID_SIZE):
    """
    :param k_pts:
    :param r_pts:
    :param im_name:
    :param w:
    :param h:
    :param grid_size:
    :return: [w, h, 3]
    """
    # get focal f
    dataset = kitti_object('/home/car/Datasets/KITTI/object')
    im_idx = int(im_name[:-4])
    calib = dataset.get_calibration(im_idx)
    f = calib.f_u    # mm -> m
    # print(f)  # 707.0493

    virtual_lidar_map = np.zeros([w, h, 4])
    grid_id2k_pt = {}

    for pt in k_pts:
        pt_2d, pt_3d, grid_id = pt[0], pt[1][:4], pt[2]
        grid_id2k_pt.update({grid_id: [pt_2d, pt_3d]})

    for pt in r_pts:
        pt_2d, grid_id = pt[0], pt[1]
        k_pt = grid_id2k_pt[tuple(grid_id)]
        k_pt_2d, k_pt_3d, k_pt_intensive = k_pt[0], k_pt[1][:3], k_pt[1][3]

        # cam2 image coord. (du, dv same with cam0 image coord.)
        du, dv = int(pt_2d[0] - k_pt_2d[0]), int(pt_2d[1] - k_pt_2d[1])

        # velo coord.
        k_pt_3d = np.expand_dims(k_pt_3d, axis=0)
        k_pt_3d_rect = calib.project_velo_to_rect(pts_3d_velo=k_pt_3d)

        k_x, k_y, k_z = k_pt_3d_rect[0]  # cam0 coord.
        dx = k_z / f * du
        dy = k_z / f * dv
        # print('real length for pixel', k_z / f)

        virtual_lidar_rect = np.array([[round(k_x+dx, 6), round(k_y+dy, 3), round(k_z, 6)]])
        # print(virtual_lidar_rect)

        # under velo coord.
        virtual_lidar_velo = calib.project_rect_to_velo(pts_3d_rect=virtual_lidar_rect)
        virtual_lidar_velo = [k_pt_3d[0][0], virtual_lidar_velo[0][1], virtual_lidar_velo[0][2], k_pt_intensive]

        virtual_lidar_map[pt_2d[0], pt_2d[1]] = virtual_lidar_velo

    return virtual_lidar_map


if __name__ == '__main__':

    instance_npz_list = os.listdir(instance_npz)
    # instance_npz_list = ['001549.npz']
    # instance_npz_list = ['005391.npz']
    for npz_name in tqdm(instance_npz_list):

        # read kfbr points
        npz_pts_name = npz_name[:-4] + '_kfb_pts.npz'
        npz_name_full = os.path.join(instance_pts_npz, npz_pts_name)
        k_points, f_points, b_points, r_points = read_info_form_kbf_pts_npz(npz_pts_name_full=npz_name_full)

        # read maps
        instance_npz_name_full = os.path.join(instance_npz, npz_name)
        im_data, im_name, im_width, im_height, k_point_num, f_point_num, b_point_num, r_point_num = \
            read_info_from_npz(instance_npz_name_full=instance_npz_name_full)
        # print(im_data.shape)    # (w, h, 18)

        print('image {} label is processing'.format(im_name))

        if r_point_num == 0:   # No r_points
            virtual_lidar_map = np.zeros([im_width, im_height, 4])
        else:
            virtual_lidar_map = get_virtual_lidar_map(k_pts=k_points, r_pts=r_points, im_name=im_name, w=im_width, h=im_height)

        # step 4: save label npz
        npz_name = im_name[0:-4] + '_label.npz'
        npz_name_full = os.path.join(instance_label, npz_name)
        np.savez(npz_name_full, virtual_lidar_map=virtual_lidar_map, im_name=im_name)

        # exit()
