"""
Annotation: this file creates instance npz files from dataset_split folder, as follow:
            package_all_map_and_save_npz(img=im, k_map=k_map, f_map=f_map, b_map=b_map, dudv_map=dudv_map,
                                     im_name=im_name, w=im_width, h=im_height,
                                     k_point_num=k_point_num, fb_point_num=fb_point_num,
                                     f_point_num=f_point_num, b_point_num=b_point_num)
Anthor: Lin, PRML Lab
Data: 20, July, 2022
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import os
import sys
import random
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from utils.util import *

from train.config import config

instance = 'Pedestrian'
GRID_SIZE = config.GRID_SIZE

SAMPLE = True
padding = config.box_padding


def get_instance_id_list(label_list):
    instance_id_list = []
    for label in label_list:
        instance_id_list.append(label[:-4])
    return instance_id_list


if __name__ == '__main__':

    npz_path = './instance_npz'
    npz_pts_path = './instance_pts_npz'

    kitti_path = './dataset_split/KITTI/object/testing'
    velo_path = kitti_path + '/velodyne'
    label_path = kitti_path + '/label_2'
    image_path = kitti_path + '/image_2'
    label_list = os.listdir(label_path)

    instance_id_list = get_instance_id_list(label_list)[:20]   # one
    # instance_id_list = ['000007', '000008', '000009']
    # instance_id_list = get_instance_id_list(label_list)        # all
    for instance_id in tqdm(instance_id_list):
        im_name = instance_id + '.png'
        print('image {} is processing'.format(im_name), im_name)
        # exit()
        ######################################
        # step 1: get pcd in image fov
        pcd_file_full = os.path.join(velo_path, instance_id + '.bin')
        pc_velo = load_velo_scan(pcd_file_full)
        pc_velo_4d = pc_velo[:, 0: 4]  # (x,y,z,r)
        pc_velo_3d = pc_velo_4d[:, 0: 3]  # (12w, 3)
        # print(pc_velo_3d.shape)
        # exit()

        #########################################
        # step 2: get foreground and background pts of instance

        # step 2-1: read 2d bbox & 3d bbox from label txt
        bboxes_2d, bboxes_3d = [], []
        label_name = instance_id + '.txt'
        label_name_full = os.path.join(label_path, label_name)
        with open(label_name_full) as f:
            rds = f.readlines()
            for rd in rds:
                info = rd.strip().split(' ')
                if info[0] == instance:
                    print(info[0])
                    bbox_2d = [int(float(info[i])) for i in range(4, 8)]
                    bboxes_2d.append(bbox_2d)
                    bbox_3d = [float(info[i]) for i in range(8, 15)]
                    bboxes_3d.append(bbox_3d)
                else:
                    continue

        # step 2-2: get pc in object fov
        dataset = kitti_object('dataset_split/KITTI/object', split='testing')
        im_idx = int(instance_id)
        calib = dataset.get_calibration(im_idx)
        im = dataset.get_image(im_idx)  # (370, 1224, 3)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_height, im_width, im_channel = im.shape

        # pts_2d, pc_fov, fov_inds = get_lidar_in_image_fov(pc_velo=pc_velo_3d, calib=calib, xmin=0,
        #                                                   ymin=0, xmax=im_width,
        #                                                   ymax=im_height,
        #                                                   clip_distance=2.0)
        # print(pc_fov.shape)
        # show_lidar(pc_fov)
        # project_points_velo2image(pc_fov, bbox_3d, im, calib, im_width, im_height, grid_size=None)
        # exit()

        pts_foreground_all, pts_background_all = [], []
        k_point_num, f_point_num, b_point_num = 0, 0, 0
        pc_fov_all = []
        for box_idx in range(len(bboxes_2d)):

            # get 2d/3d pts of one 2d_bbox in image fov, in velo coord.
            x0, y0, x1, y1 = bboxes_2d[box_idx]
            # print(x0, x1, y0, y1, im_width, im_height)

            pts_2d, pc_fov, fov_inds = get_lidar_in_image_fov(pc_velo=pc_velo_3d, calib=calib, xmin=max(0, x0-padding),
                                                              ymin=max(0, y0-padding), xmax=min(im_width-1, x1+padding),
                                                              ymax=min(im_height-1, y1+padding),
                                                              clip_distance=2.0)
            pts_reflection = pc_velo_4d[fov_inds, 3]

            # show lidar on image
            # project_points_velo2image(pc_fov, bboxes_2d[box_idx], im, calib, im_width, im_height, grid_size=None)
            # exit()

        # step 2-3: calculate 2d/3d pts in one bbox
            bbox_3d = bboxes_3d[box_idx]
            pts_2d3d_in_one_3dbbox, pts_2d3d_out_one_3dbbox, pc_instance, pc_instance_neg, fov_inds_f, fov_inds_b = \
                calculate_pts_2d3d_in_3dbbox(pts_2d=pts_2d, pc_velo=pc_fov, bbox_3d=bbox_3d, calib=calib, show_instance=False)

            if pc_instance.shape[0] == 0:
                continue
            # show_lidar(pc_instance)
            for pc in pc_instance:
                pc_fov_all.append(pc)
            project_points_velo2image(pc_instance, bboxes_2d[box_idx], im, calib, im_width, im_height)
            # project_points_velo2image(pc_instance_neg, bboxes_2d[box_idx], im, calib, im_width, im_height)
            # exit()
            continue

            pts_reflection_f = pts_reflection[fov_inds_f]
            pts_reflection_b = pts_reflection[fov_inds_b]

            pts_foreground, pts_background = pts_2d3d_in_one_3dbbox, pts_2d3d_out_one_3dbbox
            pts_foreground_num_before_sample, pts_background_num_before_sample = len(pts_foreground), len(pts_background)
            print('pts before sample: ', pts_foreground_num_before_sample, pts_background_num_before_sample)

        # step 2-4 sample pts_foreground and pts_background
            sample_num = 500
            pts_sample = True
            random.seed(1)
            if pts_sample:
                pts_foreground_new = []
                if len(pts_foreground) > sample_num:
                    sample_inds = random.sample(range(pts_foreground.__len__()), sample_num)
                    for i in sample_inds:
                        # print(pts_foreground[i], pts_reflection_f[i])
                        pts_foreground[i][1] = np.append(pts_foreground[i][1], pts_reflection_f[i])
                        pts_foreground_new.append(pts_foreground[i])
                    # pts_foreground = pts_foreground_new
                else:
                    pts_foreground_new = []
                    for i in range(pts_foreground.__len__()):
                        pts_foreground[i][1] = np.append(pts_foreground[i][1], pts_reflection_f[i])
                        pts_foreground_new.append(pts_foreground[i])

                pts_background_new = []
                # sample_num = min(pts_foreground_new.__len__(), 100)
                if len(pts_background) > sample_num:
                    sample_inds = random.sample(range(pts_background.__len__()), sample_num)
                    for i in sample_inds:
                        # print(pts_background[i], pts_reflection_f[i])
                        pts_background[i][1] = np.append(pts_background[i][1], pts_reflection_b[i])
                        pts_background_new.append(pts_background[i])
                    # pts_background = pts_background_new
                else:
                    pts_background_new = []
                    for i in range(pts_background.__len__()):
                        pts_background[i][1] = np.append(pts_background[i][1], pts_reflection_b[i])
                        pts_background_new.append(pts_background[i])

                print('Foreground points sampled: {} / {}'.format(len(pts_foreground), pts_foreground_num_before_sample))
                print('Background points sampled: {} / {}'.format(len(pts_background), pts_background_num_before_sample))

            else:
                pts_foreground_new = []
                for i in range(pts_foreground.__len__()):
                    pts_foreground[i][1] = np.append(pts_foreground[i][1], pts_reflection_f[i])
                    pts_foreground_new.append(pts_foreground[i])
                # pts_foreground = pts_foreground_new

                pts_background_new = []
                for i in range(pts_background.__len__()):
                    pts_background[i][1] = np.append(pts_background[i][1], pts_reflection_b[i])
                    pts_background_new.append(pts_background[i])
                # pts_background = pts_background_new

            pts_foreground_all = pts_foreground_all + pts_foreground_new
            pts_background_all = pts_background_all + pts_background_new

        # print(pts_foreground_all[0])  # [[406, 186], array([ 8.775,  2.39 , -0.147,  0.61 ], dtype=float32), (40, 18)]
        print('ALL Foreground points sampled: {} '.format(len(pts_foreground_all)))
        print('ALL Background points sampled: {} '.format(len(pts_background_all)))

        # show_lidar_with_3dbbox(np.array(pc_fov_all), bboxes_3d, calib)
        # obj = dataset.get_label_objects(im_idx)
        # show_image_with_boxes(img=im, objects=obj, calib=calib)
        # exit()
        # project_points_velo2image(np.array(pc_fov_all), [0, 0, 0, 0], im, calib, im_width, im_height)
        # exit()

        pts_foreground = np.array(pts_foreground_all)
        pts_background = np.array(pts_background_all)

        # pts_foreground, _, _ = refine_kf_pts(pts=pts_foreground, instance_id=instance_id, w=im_width, h=im_height)

        # step 3: get foreground and background pts of instance
        if pts_foreground.shape[0] != 0:    # Considering no foreground points in 3d bbox.
            #########################################################################################
            # step 3: create k_map
            k_map, grid_ids_pos, k_pts_2d, k_pts_3d, k_pts, pts_foreground = create_keypoint_map(
                pts_foreground=pts_foreground, w=im_width, h=im_height)
            print(pts_background.shape)

            pts_background, masks = refine_b_pts(pts=pts_background, grid_ids_pos=grid_ids_pos, instance_id=instance_id
                                                 , w=im_width, h=im_height)  # some b_points may be out of the grid

            ###############################################################################################
            # step 4: create f_map/b_map/fb_map
            # f_map, b_map = create_fb_map(pts_foreground=pts_foreground, pts_background=pts_background, w=im_width,
            #                              h=im_height)
            if pts_foreground.shape[0] == 0:      # No foreground points but key points.
                f_map = np.zeros([im_width, im_height])
            else:
                f_map = create_fb_map(pts_fb=pts_foreground, w=im_width, h=im_height)

            if pts_background.shape[0] == 0:       # No background points
                b_map = np.zeros([im_width, im_height])
            else:
                b_map = create_fb_map(pts_fb=pts_background, w=im_width, h=im_height)

            # fb_map = f_map | b_map
            fb_map = np.where(f_map + b_map != 0, 1, 0)

            ########################################
            # step 5: create dudv_map/uv_map
            # generate regression points
            uv_map, dudv_map, pts_regression = create_uv_dudv_map(k_map=k_map, f_map=f_map, fb_map=fb_map, k_pts=k_pts, w=im_width, h=im_height)
            # print(pts_regression)
            # exit()
            ######################################################################
            pts_foreground = np.array([[[0, 0], 0, 0]])
            pts_background = np.array([[[0, 0], 0, 0]])
            project_kfbr_points_velo2image2(k_pts=k_pts, f_pts=pts_foreground, b_pts=pts_background,
                                            r_pts=pts_regression, im=im, masks=masks, w=im_width, h=im_height)
            continue
            # try:
            #     pc_k = get_pc_from_kfp_pts(pts=k_pts)
            # except IndexError:
            #     pc_k = np.array([[0, 0, 0]])
            # try:
            #     pc_f = get_pc_from_kfp_pts(pts=pts_foreground)
            # except IndexError:
            #     pc_f = np.array([[0, 0, 0]])
            # try:
            #     pc_b = get_pc_from_kfp_pts(pts=pts_background)
            # except IndexError:
            #     pc_b = np.array([[0, 0, 0]])
            # try:
            #     pc_r = get_pc_from_r_pts(pts=pts_regression)
            # except IndexError:
            #     pc_r = np.array([[0, 0, 0]])
            #
            # print(pc_k.shape, pc_f.shape, pc_b.shape)
            # # pc_f = pc_rot_around_velo(pc_f, calib)
            # # pc_b = pc_rot_around_velo(pc_b, calib)
            # # pc_k = pc_rot_around_velo(pc_k, calib)
            #
            # if pc.shape[0] != 0:  # np.array([[0, 0, 0]])
            #     project_kfbr_points_velo2image(k_pc=pc_k, f_pc=np.array([[0, 0, 0]]), b_pc=np.array([[0, 0, 0]]),
            #                                    r_pc=pc_r, bbox_2d=bboxes_2d[box_idx],
            #                                    im=im, calib=calib, im_width=im_width, im_height=im_height, masks=masks)
            # exit()
            # continue
            ###############################################################################
            # Calculate point num: k_point, f_point, b_point
            k_point_num = len(k_pts_2d)
            f_point_num = pts_foreground.shape[0]
            b_point_num = pts_background.shape[0]
            r_point_num = pts_regression.shape[0]
            # print("k_point_num, fb_point_num, f_point_num, b_point_num: {} {} {} {}"
            #       .format(k_point_num, fb_point_num, f_point_num, b_point_num))

        else:       # No key pointd
            k_map = np.zeros([im_width, im_height])
            f_map = k_map
            b_map = k_map
            dudv_map = np.zeros([im_width, im_height, 2])
            fb_map = k_map
            uv_map = k_map

            k_pts = np.zeros([k_point_num, 4])
            f_pts = k_pts
            b_pts = k_pts

            k_point_num = 0
            f_point_num = 0
            b_point_num = 0
            r_point_num = 0

        # save kfbr points as npz
        if k_point_num == 0:
            k_pts = np.zeros([k_point_num, 4])
            f_pts = k_pts
            b_pts = k_pts
            r_pts = k_pts

        else:
            k_pts = k_pts
            f_pts = pts_foreground
            b_pts = pts_background
            r_pts = pts_regression

        # calculate min cut area
        bboxes_2d = np.array(bboxes_2d)
        min_cut_x = max(min(bboxes_2d[:, 0]) - padding, 0)
        min_cut_y = max(min(bboxes_2d[:, 1]) - padding, 0)
        max_cut_x = min(max(bboxes_2d[:, 2]) + padding, im_width)
        max_cut_y = min(max(bboxes_2d[:, 3]) + padding, im_height)
        # print(bboxes_2d)
        cut_area = [min_cut_x, min_cut_y, max_cut_x, max_cut_y]

        # print(k_pts.shape, f_pts.shape, b_pts.shape)
        save_kfb_pts_npz(im_name=im_name, k_pts=k_pts, f_pts=f_pts, b_pts=b_pts, r_pts=r_pts, save_folder=npz_pts_path)
        # print(k_pts.shape, f_pts.shape, b_pts.shape, r_pts.shape)

        # create pts_map: k_pts, f_pts, b_pts
        k_pts_map = create_pts_map(kfb_pts=k_pts, w=im_width, h=im_height)
        f_pts_map = create_pts_map(kfb_pts=f_pts, w=im_width, h=im_height)
        b_pts_map = create_pts_map(kfb_pts=b_pts, w=im_width, h=im_height)

        # package all map as npz
        package_all_map_and_save_npz(img=im, k_map=k_map, f_map=f_map, b_map=b_map, fb_map=fb_map,
                                     dudv_map=dudv_map, uv_map=uv_map,
                                     k_pts_map=k_pts_map, f_pts_map=f_pts_map, b_pts_map=b_pts_map,
                                     im_name=im_name, w=im_width, h=im_height,
                                     k_point_num=k_point_num, r_point_num=r_point_num,
                                     f_point_num=f_point_num, b_point_num=b_point_num,
                                     cut_area=cut_area,
                                     save_folder=npz_path)





