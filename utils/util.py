import os
import sys
import numpy

import random
random.seed(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
from kitti.kitti_object import kitti_object, show_lidar_on_image, show_lidar_with_2dbbox_on_image, show_image_with_boxes

sys.path.append('mayavi')
# sys.path.append(os.path.dirname('mayavi'))
import mayavi.mlab as mlab
from viz_util import draw_lidar, draw_gt_boxes3d

import numpy as np
import cv2
import matplotlib.pyplot as plt

from train.config import config
GRID_SIZE = config.GRID_SIZE
MAX_DISTANCE = config.MAX_DISTANCE

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def show_lidar_with_3dbbox(pcd, bboxes, calib):

    fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=None, engine=None, size=(800, 500))
    draw_lidar(pcd, fig=fig)

    for bbox in bboxes:
        h, w, l = bbox[0:3]
        x, y, z = bbox[3:6]
        ry = bbox[6]

        # step 1: cam_loc -> cam
        # coordinate change: rotation
        R = roty(ry)
        # 3d bounding box corners
        x_corners = [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]
        y_corners = [-h, -h, -h, -h, 0, 0, 0, 0]
        z_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

        # coordinate change: translate
        corners_3d[0, :] = corners_3d[0, :] + x
        corners_3d[1, :] = corners_3d[1, :] + y
        corners_3d[2, :] = corners_3d[2, :] + z

        # print(np.zeros_like(corners_3d[0, :]))
        corners_4d = np.append(corners_3d, [np.zeros_like(corners_3d[0, :])], axis=0)

        # step 2: cam -> velo
        velo2cam = calib.V2C
        R = velo2cam[:3, :3]
        T = velo2cam[:3, 3]
        # print(R, T)

        R1 = np.linalg.inv(R)
        T8 = np.zeros([3, 8])

        for i in range(8):
            T8[:, i] = T

        K = corners_3d - T8
        corners_velo = np.dot(R1, np.vstack([K[0, :], K[1, :], K[2, :]]))

        top_p0_velo = corners_velo[:, 0]
        top_p1_velo = corners_velo[:, 1]
        top_p2_velo = corners_velo[:, 2]
        top_p3_velo = corners_velo[:, 3]
        down_p0_velo = corners_velo[:, 4]
        down_p1_velo = corners_velo[:, 5]
        down_p2_velo = corners_velo[:, 6]
        down_p3_velo = corners_velo[:, 7]

        # plot 3d bbox
        mlab.plot3d([top_p0_velo[0], top_p1_velo[0]], [top_p0_velo[1], top_p1_velo[1]],
                    [top_p0_velo[2], top_p1_velo[2]])
        mlab.plot3d([top_p1_velo[0], top_p2_velo[0]], [top_p1_velo[1], top_p2_velo[1]],
                    [top_p1_velo[2], top_p2_velo[2]])
        mlab.plot3d([top_p2_velo[0], top_p3_velo[0]], [top_p2_velo[1], top_p3_velo[1]],
                    [top_p2_velo[2], top_p3_velo[2]])
        mlab.plot3d([top_p3_velo[0], top_p0_velo[0]], [top_p3_velo[1], top_p0_velo[1]],
                    [top_p3_velo[2], top_p0_velo[2]])

        mlab.plot3d([down_p0_velo[0], down_p1_velo[0]], [down_p0_velo[1], down_p1_velo[1]],
                    [down_p0_velo[2], down_p1_velo[2]])
        mlab.plot3d([down_p1_velo[0], down_p2_velo[0]], [down_p1_velo[1], down_p2_velo[1]],
                    [down_p1_velo[2], down_p2_velo[2]])
        mlab.plot3d([down_p2_velo[0], down_p3_velo[0]], [down_p2_velo[1], down_p3_velo[1]],
                    [down_p2_velo[2], down_p3_velo[2]])
        mlab.plot3d([down_p3_velo[0], down_p0_velo[0]], [down_p3_velo[1], down_p0_velo[1]],
                    [down_p3_velo[2], down_p0_velo[2]])

        mlab.plot3d([top_p0_velo[0], down_p0_velo[0]], [top_p0_velo[1], down_p0_velo[1]],
                    [top_p0_velo[2], down_p0_velo[2]])
        mlab.plot3d([top_p1_velo[0], down_p1_velo[0]], [top_p1_velo[1], down_p1_velo[1]],
                    [top_p1_velo[2], down_p1_velo[2]])
        mlab.plot3d([top_p2_velo[0], down_p2_velo[0]], [top_p2_velo[1], down_p2_velo[1]],
                    [top_p2_velo[2], down_p2_velo[2]])
        mlab.plot3d([top_p3_velo[0], down_p3_velo[0]], [top_p3_velo[1], down_p3_velo[1]],
                    [top_p3_velo[2], down_p3_velo[2]])

    mlab.show(1)
    raw_input()


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax, clip_distance=2.0):

    pts_2d = calib.project_velo_to_image(pc_velo)  # velo coord. -> camera coord.
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
               (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:, 0]>clip_distance)   # filter those point cloud with x < 2.0
    imgfov_pc_velo = pc_velo[fov_inds, :]
    pts_2d = pts_2d[fov_inds, :]
    return pts_2d, imgfov_pc_velo, fov_inds


def show_2d_bbox(im_name, bboxes):
    kitti_path = './dataset_split/KITTI/object/training'
    im_name_full = kitti_path + '/image_2/' + im_name
    im = cv2.imread(im_name_full)

    for bbox in bboxes:
        p_tl, p_dr = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        im = cv2.rectangle(im, p_tl, p_dr, (0, 255, 0), 2)
    cv2.imshow(im_name, im)
    cv2.waitKey(0)


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def project_points_velo2image(pcd, bbox_2d, im, calib, im_width, im_height, grid_size=None):
    # show_lidar_on_image(pcd, im, calib, im_width, im_height, showtime=True, grid_size=None)
    show_lidar_with_2dbbox_on_image(pcd, bbox_2d, im, calib, im_width, im_height, showtime=True, grid_size=grid_size)


def project_points_add_velo2image(pcd, bbox_2d, im, calib, im_width, im_height, masks, pc_add, grid_size=None):
    # show_lidar_on_image(pcd, im, calib, im_width, im_height, showtime=True, grid_size=None)
    show_lidar_with_2dbbox_on_image(pcd, bbox_2d, im, calib, im_width, im_height, masks, pc_add, showtime=True, grid_size=None)


def project_kfbr_points_velo2image(k_pc, f_pc, b_pc, r_pc, bbox_2d, im, calib, im_width, im_height, masks):
    """
    :param k_pc: 3d
    :param f_pc: 3d
    :param b_pc: 3d
    :param r_pc: 2d
    :param bbox_2d:
    :param im:
    :param calib:
    :param im_width:
    :param im_height:
    :param masks:
    :return:
    """

    pts_color = [(255, 0, 0), [255, 128, 0], [128, 128, 128], [255, 255, 0]]
    for i, pcd in enumerate([k_pc[:, 0:3], f_pc[:, 0:3], b_pc[:, 0:3]]):
        img = show_lidar_with_2dbbox_on_image(pcd, bbox_2d, im, calib, im_width, im_height, masks,
                                              showtime=True, pts_color=pts_color[i], grid_size=GRID_SIZE)
    for i in range(r_pc.shape[0]):
        cv2.circle(img, (int(r_pc[i, 0]), int(r_pc[i, 1])), 1, color=pts_color[3], thickness=-1)
    while True:
        cv2.imshow('im', img)
        ch = cv2.waitKey(1)
        if ch == ord('q') or ch == ord('Q'):
            break


def project_kfbr_points_velo2image2(k_pts, f_pts, b_pts, r_pts, im, masks, w, h, grid_size=GRID_SIZE):
    """
    :param k_pc: 3d
    :param f_pc: 3d
    :param b_pc: 3d
    :param r_pc: 2d
    :param bbox_2d:
    :param im:
    :param calib:
    :param im_width:
    :param im_height:
    :param masks:
    :return:
    """
    if k_pts.shape[0] == 0:
        k_pts = np.array([[0, 0, 0]])
    if f_pts.shape[0] == 0:
        f_pts = np.array([[0, 0, 0]])
    if b_pts.shape[0] == 0:
        b_pts = np.array([[0, 0, 0]])
    if r_pts.shape[0] == 0:
        r_pts = np.array([[0, 0, 0]])

    if masks is not None:
        for i in range(masks.shape[2]):
            im[:, :, 0] = np.where(masks[:, :, i] == 1, 128, im[:, :, 0])

    if grid_size is not None:
        cell_id = (int(w/grid_size), int((h/grid_size)))
        for id0 in range(1, cell_id[0]+1):
            cv2.line(im, (id0 * grid_size, 0), (id0 * grid_size, h), color=(255, 0, 0))
        for id1 in range(1, cell_id[1]+1):
            cv2.line(im, (0, id1 * grid_size), (w, id1 * grid_size), color=(255, 0, 0))

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pts_color = [(0, 0, 255), [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    for i, pts_2d in enumerate([k_pts[:, 0], f_pts[:, 0], b_pts[:, 0], r_pts[:, 0]]):

        for pt in pts_2d:
            im = cv2.circle(im, (int(pt[0]), int(pt[1])), 1, color=pts_color[i], thickness=-1)

        while True:
            cv2.imshow('im', im)
            ch = cv2.waitKey(1)
            if ch == ord('q') or ch == ord('Q'):
                break


def calculate_pts_2d3d_in_3dbbox(pts_2d, pc_velo, bbox_3d, calib, grid_size=GRID_SIZE, show_instance=True):
    """
    :param pts_2d:
    :param pc_velo:
    :param bbox_3d:
    :param calib:
    :param show_instance:
    :return: [[pts_2d, pts_3d],...]
    """

    pts_3d_in_one_3dbbox = []
    pts_3d_out_one_3dbbox = []

    pts_2d3d_in_one_3dbbox = []
    pts_2d3d_out_one_3dbbox = []

    velo2cam0 = calib.V2C  # shape [3, 4]
    velo2cam0 = np.append(velo2cam0, [[0, 0, 0, 1]], axis=0)

    fov_inds_f, fov_inds_b = [], []
    for i, p_velo in enumerate(pc_velo):

        # step 1: transform points from velo to cam
        p_velo_4 = np.append(p_velo, 1)  # 3 -> 4
        p_cam = np.dot(velo2cam0, p_velo_4)
        p_cam = p_cam[0:3]

        # step 2: transform points from cam to cam_loc
        h, w, l = bbox_3d[0:3]
        x, y, z = bbox_3d[3:6]
        ry = bbox_3d[6]

        R = roty(ry)
        R1 = np.linalg.inv(R)
        T = np.array([x, y, z])
        p_cam_loc = np.dot(R1, np.transpose(p_cam - T))

        # step 3: calculate_points_in_3d_bbox
        x_min, x_max = -l / 2, l / 2
        z_min, z_max = -w / 2, w / 2
        y_min, y_max = -h, 0

        if (p_cam_loc[0] <= x_max) & (p_cam_loc[0] >= x_min) & \
                (p_cam_loc[1] <= y_max) & (p_cam_loc[1] >= y_min) & \
                (p_cam_loc[2] <= z_max) & (p_cam_loc[2] >= z_min):

            pts_3d_in_one_3dbbox.append(p_velo)
            pt_2d = [int(xy) for xy in pts_2d[i]]
            grid_pos_idx = (int(pt_2d[0] / grid_size), int(pt_2d[1] / grid_size))
            pts_2d3d_in_one_3dbbox.append([pt_2d, p_velo, grid_pos_idx])

            fov_inds_f.append(True)
            fov_inds_b.append(False)

        else:
            pts_3d_out_one_3dbbox.append(p_velo)
            pt_2d = [int(xy) for xy in pts_2d[i]]
            grid_pos_idx = (int(pt_2d[0] / grid_size), int(pt_2d[1] / grid_size))
            pts_2d3d_out_one_3dbbox.append([pt_2d, p_velo, grid_pos_idx])
            fov_inds_f.append(False)
            fov_inds_b.append(True)

    if show_instance:
        pcd = np.array(pts_3d_in_one_3dbbox)
        show_lidar_with_3dbbox(pcd=pcd, bboxes=[bbox_3d], calib=calib)

    return pts_2d3d_in_one_3dbbox, pts_2d3d_out_one_3dbbox, np.array(pts_3d_in_one_3dbbox), np.array(pts_3d_out_one_3dbbox), fov_inds_f, fov_inds_b


def get_fb_lidar_in_image_fov(pc_velo, calib):
    return calib.project_velo_to_image(pc_velo)


# def create_keypoint_map(pts_foreground, w, h, grid_size=GRID_SIZE):
#     """
#     :param pts_2d: 2d foreground points
#     :param w:
#     :param h:
#     :param grid_size: default = 20
#     :return: k_map, grid_ids_pos, k_pts_2d, k_pts_3d, k_pts, new_pts_foreground
#     """
#     k_map = np.zeros([w, h])
#     pts_dict = {}
#     pts_2d = pts_foreground[:, 0]
#     pts_3d = pts_foreground[:, 1]
#
#     grid_pos_idx = pts_foreground[:, 2]
#
#     # create pts_dict, key: grid_idx, value: [[pt_2d, pts_3d], ...]
#     for i, grid_idx in enumerate(grid_pos_idx):
#         try:
#             pts_dict[grid_idx].append([pts_2d[i], pts_3d[i]])
#         except KeyError:
#             pts_dict.update({grid_idx: [[pts_2d[i], pts_3d[i]]]})
#
#     grid_ids_pos = []
#     k_pts_2d, k_pts_3d = [], []
#     k_pts = []
#     new_pts_foreground = []
#
#     for key, points in pts_dict.items():
#         points = np.array(points, dtype=object)
#
#         pts_2d = points[:, 0]
#         pts_3d = points[:, 1]
#
#         k_pt_2d_x, k_pt_2d_y = pts_2d[-1][0], pts_2d[-1][1]
#         k_pt_2d = [k_pt_2d_x, k_pt_2d_y]
#         k_map[k_pt_2d_x, k_pt_2d_y] = 1  # select the last point in this grid as k-point.
#
#         k_pt_3d = pts_3d[-1]
#
#         k_pts_2d.append(k_pt_2d)
#         k_pts_3d.append(k_pt_3d)
#         k_pts.append([k_pt_2d, k_pt_3d, key])
#
#         grid_ids_pos.append(key)
#
#         # get new foreground points
#         if points.shape[0] > 1:
#             for pt in points[0:-1]:
#                 pt = list(pt)
#                 pt.append(key)
#                 new_pts_foreground.append(pt)
#
#     # print(len(new_pts_foreground))
#
#     k_pts = np.array(k_pts, dtype=object)
#     new_pts_foreground = np.array(new_pts_foreground, dtype=object)
#     return k_map, grid_ids_pos, k_pts_2d, k_pts_3d, k_pts, new_pts_foreground


def create_keypoint_map(pts_foreground, w, h, grid_size=GRID_SIZE):
    """
    :param pts_2d: 2d foreground points
    :param w:
    :param h:
    :param grid_size: default = 20
    :return: k_map, grid_ids_pos, k_pts_2d, k_pts_3d, k_pts, new_pts_foreground
    """
    k_map = np.zeros([w, h])
    pts_dict = {}
    pts_2d = pts_foreground[:, 0]
    pts_3d = pts_foreground[:, 1]

    grid_pos_idx = pts_foreground[:, 2]

    # create pts_dict, key: grid_idx, value: [[pt_2d, pts_3d], ...]
    for i, grid_idx in enumerate(grid_pos_idx):
        try:
            pts_dict[grid_idx].append([pts_2d[i], pts_3d[i]])
        except KeyError:
            pts_dict.update({grid_idx: [[pts_2d[i], pts_3d[i]]]})

    grid_ids_pos = []
    k_pts_2d, k_pts_3d = [], []
    k_pts = []
    new_pts_foreground = []

    for grid_id, points in pts_dict.items():

        grid_center = np.array([[grid_id[0] * GRID_SIZE + GRID_SIZE / 2, grid_id[1] * GRID_SIZE + GRID_SIZE / 2]])
        pts_2d_one_grid = np.array(pts_dict[grid_id])[:, 0]
        pts_2d_one_grid = np.array([np.array(p) for p in pts_2d_one_grid])
        # print(grid_center, pts_2d_one_grid)

        bias = pts_2d_one_grid - grid_center  # Nx2 - 1x2
        bias_manhattan = np.linalg.norm(bias, ord=1, axis=1)
        k_idx = np.argmin(bias_manhattan)   # select the most center point in this grid as k-point.
        # print(k_idx)
        # exit()

        points = np.array(points, dtype=object)
        pts_2d = points[:, 0]
        pts_3d = points[:, 1]

        k_pt_2d = pts_2d[k_idx]
        k_map[k_pt_2d[0], k_pt_2d[1]] = 1

        k_pt_3d = pts_3d[k_idx]

        k_pts_2d.append(k_pt_2d)
        k_pts_3d.append(k_pt_3d)
        k_pts.append([k_pt_2d, k_pt_3d, grid_id])

        grid_ids_pos.append(grid_id)

        # get new foreground points
        if points.shape[0] > 1:
            for pt in points[0:k_idx]:
                pt = list(pt)
                pt.append(grid_id)
                new_pts_foreground.append(pt)
            if k_idx != points.shape[0] - 1:
                for pt in points[k_idx+1:]:
                    pt = list(pt)
                    pt.append(grid_id)
                    new_pts_foreground.append(pt)


    # print(len(new_pts_foreground))

    k_pts = np.array(k_pts, dtype=object)
    new_pts_foreground = np.array(new_pts_foreground, dtype=object)
    return k_map, grid_ids_pos, k_pts_2d, k_pts_3d, k_pts, new_pts_foreground



def create_fb_map(pts_fb, w, h):
    f_map = np.zeros([w, h])

    pts_2d_fb = pts_fb[:, 0]
    pts_3d_fb = pts_fb[:, 1]

    for p in pts_2d_fb:
        w_idx, h_idx = p[0], p[1]
        f_map[w_idx, h_idx] = 1          # create f_map, b_map

    return f_map


def get_fb_pts_2d(pts_fb, grid_id=None, grid_size=GRID_SIZE):
    pts_dict = {}
    pts_2d = pts_fb[:, 0]
    pts_3d = pts_fb[:, 1]
    grid_pos_idx = pts_fb[:, 2]

    # print('pts_foreground1', pts_foreground.shape[0])

    if grid_id:
        for i, grid_id in enumerate(grid_pos_idx):
            try:
                pts_dict[grid_id].append([pts_2d[i], pts_3d[i]])
            except KeyError:
                pts_dict.update({grid_id: [[pts_2d[i], pts_3d[i]]]})

    return pts_dict


def create_dudv_map(pts_regression_dict, k_points, w, h, grid_size=GRID_SIZE):
    dudv_map = np.zeros([w, h, 2])
    uv_map = np.zeros([w, h])

    for k_point in k_points:
        k_point_grid_id = k_point[2]
        # print(k_point_grid_id)
        try:
            pts_regression_in_grid = pts_regression_dict[k_point_grid_id]
        except KeyError:
            pts_regression_in_grid = []
            # print('- no foreground point in {}'.format(k_point_grid_id))

        if pts_regression_in_grid is None:
            pass
        for pt_grid in pts_regression_in_grid:
            pt_2d, pt_3d = pt_grid[0], pt_grid[1]
            k_pt_2d = k_point[0]
            du, dv = pt_2d[0] - k_pt_2d[0], pt_2d[1] - k_pt_2d[1]
            # print('--f',  du, dv)
            dudv_map[pt_2d[0], pt_2d[1], 0] = du
            dudv_map[pt_2d[0], pt_2d[1], 1] = dv
            uv_map[pt_2d[0], pt_2d[1]] = 1

    return dudv_map, uv_map


def show_map(i_map):
    k_map = np.transpose(i_map)
    plt.imshow(k_map, cmap='gray')
    plt.show()


def package_all_map_and_save_npz(img, k_map, f_map, b_map, fb_map, dudv_map, uv_map, k_pts_map, f_pts_map, b_pts_map,
                                 im_name, w, h, k_point_num, r_point_num, f_point_num, b_point_num, cut_area,
                                 save_folder='./instance_npz/'):
    # print('img:', img.shape)        # (h, w, 3)
    # print('k_map', k_map.shape)     # (w, h)
    # print('f_map:', f_map.shape)    # (w, h)
    # print('b_map:', b_map.shape)    # (w, h)
    # print('dudv_map:', dudv_map.shape)  # (w, h, 2)

    img = img.transpose([1, 0, 2])
    k_map = np.expand_dims(k_map, axis=2)
    f_map = np.expand_dims(f_map, axis=2)
    b_map = np.expand_dims(b_map, axis=2)
    fb_map = np.expand_dims(fb_map, axis=2)
    uv_map = np.expand_dims(uv_map, axis=2)

    data = np.concatenate((img,         # 3
                           k_map,       # 1
                           k_pts_map,   # 4
                           f_map,       # 1
                           f_pts_map,   # 4
                           b_map,       # 1
                           b_pts_map,   # 4
                           uv_map,      # 1
                           dudv_map,    # 2
                           ), axis=2)
    """
    img: [w, h, 3]
    k_map: [w, h, 1]
    f_map: [w, h, 1]
    b_map: [w, h, 1]
    dudv_map: [w, h, 2]
    uv_map: [w, h, 1]
    k_pts_map: [w, h, 4]
    b_pts_map: [w, h, 4]
    f_pts_map: [w, h, 4]
    --------------------
    all_map: [w, h, 21]
    """
    print('uv_map', data[:, :, 15].sum())
    data_cut = data[cut_area[0]:cut_area[2], cut_area[1]:cut_area[3], :]
    # print(data_cut.shape)
    # exit()
    npz_name = im_name[0:-4] + '.npz'
    npz_name_full = os.path.join(save_folder, npz_name)
    np.savez(npz_name_full, data=data_cut, im_name=im_name, im_width=w, img_height=h, k_point_num=k_point_num,
             r_point_num=r_point_num, f_point_num=f_point_num, b_point_num=b_point_num, cut_area=cut_area)
    # print('image {} processed'.format(im_name), im_name)


def show_lidar(pcd):
    pcd = pcd[:, :3]
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(pcd, fig=fig)

    mlab.show(1)
    raw_input()


def read_info_from_npz(instance_npz_name_full):
    info = np.load(instance_npz_name_full)
    # print(info.files)
    # exit()
    im_data = info['data']
    im_name = info['im_name']
    im_width = info['im_width']
    im_height = info['img_height']
    k_point_num = info['k_point_num']
    f_point_num = info['f_point_num']
    b_point_num = info['b_point_num']
    r_point_num = info['r_point_num']

    return im_data, str(im_name), im_width, im_height, k_point_num, f_point_num, b_point_num, r_point_num


def draw_2d_pts_on_image(pts_2d, im, save_name='im.png'):
    for pt in pts_2d:
        color = [255, 0, 0]
        cv2.circle(im, (round(pt[0]), round(pt[1])), 1, color=tuple(color), thickness=-1)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_name, im)
    # cv2.imshow('im', im)
    # cv2.waitKey(0)


def read_mask(npz_name_full):
    info = np.load(npz_name_full, allow_pickle=True)
    # print(info.files)  #  ['bboxes', 'masks', 'classes', 'scores']
    classes = info['classes']
    # print(classes)

    # only save person
    r_mask = None
    for i, cls in enumerate(classes):
        if cls == 'person':
            if r_mask is not None:
                r_mask = np.concatenate([r_mask, np.expand_dims(info['masks'][:, :, i], 2)], axis=2)
            else:
                r_mask = info['masks'][:, :, i]
                r_mask = np.expand_dims(r_mask, axis=2)
    if r_mask is None:
        r_mask = np.zeros_like(info['masks'][:, :, 0:1])
    return r_mask


def read_mask2(npz_name_full, w, h):
    """
    :param npz_name_full:
    :return: maskrcnn_labelling for train,but no testing mask
             https://github.com/HeylenJonas/KITTI3D-Instance-Segmentation-Devkit
    """
    try:
        info = np.load(npz_name_full, allow_pickle=True)
        r_mask = info['mask']
    except FileNotFoundError:
        r_mask = np.zeros([h, w])   # r_mask: [h, w]

    # if r_mask.sum() !=0:
    #     print(np.unique(r_mask))
    #     exit()
    r_mask = np.expand_dims(r_mask, axis=2)
    return r_mask


def refine_b_pts(pts, grid_ids_pos, instance_id, w, h):
    """
    :param pts:
    :param grid_ids_pos:
    :param grid_size:
    :return:
    """
    mask_path = './KITTI_MASK/testing'
    mask_name = instance_id + '.npz'
    mask_name_full = os.path.join(mask_path, mask_name)
    r_mask = read_mask2(mask_name_full, w, h)
    # print(r_mask.shape)

    refined_pts = []
    for i, pt in enumerate(pts):
        pt_flag = True   # background point： true
        pt_2d, _, pt_2d_grid_id = pt[0], pt[1], pt[2]

        grid_ids_pos = np.array(grid_ids_pos)
        pt_2d_grid_id = np.array(pt_2d_grid_id)

        for i in range(r_mask.shape[2]):
            mask = r_mask[:, :, i]
            if mask[pt_2d[1], pt_2d[0]]:    # person
                pt_flag = False
        if pt_flag and (grid_ids_pos == pt_2d_grid_id).all(1).any():
            refined_pts.append(pt)

    return np.array(refined_pts, dtype=object), r_mask


def refine_kf_pts(pts, instance_id, w, h):

    # mask_path = '/home/car/exps/maskrcnn-kitti-labelling/out'
    mask_path = './KITTI_MASK/testing'
    mask_name = instance_id + '.npz'
    mask_name_full = os.path.join(mask_path, mask_name)
    r_mask = read_mask2(mask_name_full, w, h)
    print(r_mask.shape)

    refined_pts = []
    refined_pts_neg = []
    for i, pt in enumerate(pts):
        pt_flag = False   # kf point： true
        pt_2d, _, pt_2d_grid_id = pt[0], pt[1], pt[2]

        for i in range(r_mask.shape[2]):
            mask = r_mask[:, :, i]
            if mask[pt_2d[1], pt_2d[0]]:    # person
                pt_flag = True
        if pt_flag:
            refined_pts.append(pt)   # kf points selected
            # pass
        else:
            refined_pts_neg.append(pt)      # kf points not selected
            pass

    return np.array(refined_pts, dtype=object), r_mask, refined_pts_neg


def save_kfb_pts_npz(im_name, k_pts, f_pts, b_pts, r_pts, save_folder):
    # kfb_pts_dict = {}
    # kfb_pts_dict.update({'k_points': k_pts})
    # kfb_pts_dict.update({'f_points': f_pts})
    # kfb_pts_dict.update({'b_points': b_pts})

    npz_name = im_name[0:-4] + '_kfb_pts.npz'
    npz_name_full = os.path.join(save_folder, npz_name)
    np.savez(npz_name_full, k_points=k_pts, f_points=f_pts, b_points=b_pts, r_points=r_pts)


def create_pts_map(kfb_pts, w, h):
    kfb_pts_map = np.zeros([w, h, 4])
    # print(kfb_pts.shape)
    if kfb_pts.shape[0] == 0:
        return kfb_pts_map
    pts_2d = kfb_pts[:, 0]
    pts_3d = kfb_pts[:, 1]

    for i, pt_2d in enumerate(pts_2d):
        u, v = pt_2d[0], pt_2d[1]
        kfb_pts_map[u, v] = pts_3d[i]
        # print(pts_3d[i])
        # exit()

    # print('kbf point in kbf_map: {}'.format(i+1))
    return kfb_pts_map


def split_regression_pts_from_fb_pts(f_pts, b_pts, interval=3):
    regression_pts, f_pts_, b_pts_ = [], [], []
    for i, pt in enumerate(f_pts):
        if i % interval == 0:
            pt = list(pt)
            pt.append('f')
            regression_pts.append(pt)
        else:
            f_pts_.append(pt)

    for i, pt in enumerate(b_pts):
        if i % interval == 0:
            pt = list(pt)
            pt.append('b')
            regression_pts.append(pt)
        else:
            b_pts_.append(pt)

    regression_pts = np.array(regression_pts, dtype=object)
    f_pts_ = np.array(f_pts_, dtype=object)
    b_pts_ = np.array(b_pts_, dtype=object)

    return regression_pts, f_pts_, b_pts_


def get_pc_from_kfp_pts(pts):
    pc = pts[:, 1]
    pc = [list(p) for p in pc]
    # exit()
    return numpy.array(pc)


def get_pc_from_r_pts(pts):
    pc = pts[:, 0]
    pc = [list(p) for p in pc]
    # exit()
    return numpy.array(pc)


def pc_rot_around_velo(pc_velo, calib):
    pc_rect = calib.project_velo_to_rect(pc_velo[:, 0:3])
    pc_new = []
    for i in range(pc_velo.shape[0]):
        tan_velo = -pc_velo[i][0]/pc_velo[i][1]
        tan_rect = pc_rect[i][2]/pc_rect[i][0]
        angle_velo = np.arctan(tan_velo)
        angle_rect = np.arctan(tan_rect)
        rot = angle_velo - angle_rect
        # print(angle_velo, angle_rect, rot)
        d = np.sqrt(pc_velo[i][0] ** 2 + pc_velo[i][1] ** 2)
        rot /= 5
        # rot = 0
        if tan_velo > 0:
            x_velo = d * np.sin(angle_velo + rot)
            y_velo = -d * np.cos(angle_velo + rot)
            # delta_y = pc_velo[i][0] * np.tan(rot)
            # x_velo = pc_velo[i][0]
            # y_velo = pc_velo[i][1] + delta_y
        else:
            x_velo = -d * np.sin(angle_velo + rot)
            y_velo = d * np.cos(angle_velo + rot)
            # delta_y = pc_velo[i][0] * np.tan(rot)
            # x_velo = pc_velo[i][0]
            # y_velo = pc_velo[i][1] + delta_y

        # print(pc_velo[i], [x_velo, y_velo, pc_velo[i][2]])
        pc_new.append([x_velo, y_velo, pc_velo[i][2]])

    return np.array(pc_new)


def create_uv_dudv_map(k_map, f_map, fb_map, k_pts, w, h, N=config.VP_NUM, M=config.GRID_POINT_NUM, GRID_SIZE=config.GRID_SIZE):
    """
    :param k_map:
    :param fb_map:
    :param k_pts:
    :param w:
    :param h:
    :param N:
    :param GRID_SIZE:
    :return: uv_map, dudv_map
    if pts in grid more than M, do less dudv.
    """
    uv_map = np.zeros([w, h])
    dudv_map = np.zeros([w, h, 2])

    f_ids = np.transpose(np.nonzero(f_map))
    grid_num = {}
    for f_id in f_ids:
        grid_id = (int(f_id[0] / GRID_SIZE), int(f_id[1] / GRID_SIZE))
        try:
            grid_num.update({grid_id: grid_num[grid_id]+1})
        except KeyError:
            grid_num.update({grid_id: 1})
    # print(grid_num)
    # exit()
    kfb_map = np.where(k_map + fb_map > 0, 0, 1)   # 0: occupy, 1: free
    r_pts = []
    for pt in k_pts:
        pt_k, grid_id = pt[0], pt[2]
        print(grid_id)
        print(grid_id[0]-1, grid_id[0])
        surrounding_grid_point_num = 0
        try:
            self_grid_num = grid_num[grid_id]   # middle grid(self)
            surrounding_grid_point_num += self_grid_num
            # print('-0', self_grid_num)
        except KeyError:
            surrounding_grid_point_num += 0
        try:
            left_grid_num = grid_num[(grid_id[0]-1, grid_id[1])]   # left grid
            surrounding_grid_point_num += left_grid_num
            print('-1', left_grid_num)
        except KeyError:
            surrounding_grid_point_num += 0
        try:
            top_grid_num = grid_num[(grid_id[0], grid_id[1]-1)]   # top grid
            surrounding_grid_point_num += top_grid_num
            print('-2', top_grid_num)
        except KeyError:
            surrounding_grid_point_num += 0
        try:
            right_grid_num = grid_num[(grid_id[0]+1, grid_id[1])]   # right grid
            surrounding_grid_point_num += right_grid_num
            print('-3', right_grid_num)
        except KeyError:
            surrounding_grid_point_num += 0
        try:
            down_grid_num = grid_num[(grid_id[0], grid_id[1]+1)]   # down grid
            surrounding_grid_point_num += down_grid_num
            print('-4', down_grid_num)
        except KeyError:
            surrounding_grid_point_num += 0

        print(surrounding_grid_point_num)
        if surrounding_grid_point_num > M:
            continue

        u_id, v_id = grid_id

        patch = np.ones([GRID_SIZE, GRID_SIZE])

        patch_kfb = kfb_map[GRID_SIZE*grid_id[0]: GRID_SIZE*(grid_id[0]+1),
                            GRID_SIZE*grid_id[1]: GRID_SIZE*(grid_id[1]+1)]
        # assert patch.shape == patch_kfb
        try:    # in case of some bounder issue
            patch_free = patch * patch_kfb
        except ValueError:
            patch = patch[0:patch_kfb.shape[0], 0:patch_kfb.shape[1]]
            patch_free = patch * patch_kfb
        patch_free_ids = np.transpose(np.nonzero(patch_free))

        # method 1: generate vp randomly
        # sample_id = random.sample(list(patch_free_ids), N)
        # pt_k_grid_x, pt_k_grid_y = pt_k[0] % GRID_SIZE, pt_k[1] % GRID_SIZE

        # method 2: generate vp horizontally
        sample_id = []
        pt_k_grid_x, pt_k_grid_y = pt_k[0] % GRID_SIZE, pt_k[1] % GRID_SIZE
        for p in patch_free_ids:
            if p[1] == pt_k_grid_y and p[0] % 2 == 1:   # horizontal/h
                sample_id.append(p)
            if p[1] == (pt_k_grid_y - 3) and p[0] % 2 == 1:  # vertical
                sample_id.append(p)
            if p[1] == (pt_k_grid_y + 3) and p[0] % 2 == 1:   # vertical
                sample_id.append(p)
        # print(pt_k)
        # print(pt_k_grid_x, pt_k_grid_y)
        # print(sample_id)

        for s_id in sample_id:
            du = s_id[0] - pt_k_grid_x
            dv = s_id[1] - pt_k_grid_y
            uv_map[pt_k[0]+du, pt_k[1]+dv] = 1
            dudv_map[pt_k[0] + du, pt_k[1] + dv, 0] = du
            dudv_map[pt_k[0] + du, pt_k[1] + dv, 1] = dv
            # print(grid_id)
            r_pts.append([[pt_k[0] + du, pt_k[1] + dv], grid_id])
            # print(pt_k[0] + du, pt_k[1] + dv)
        # exit()
    print('VP_NUM', uv_map.sum())
    r_pts = np.array(r_pts, dtype=object)
    return uv_map, dudv_map, r_pts


# if __name__ == '__main__':
#     pts1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     pts2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#     _, _, _ = split_regression_pts_from_fb_pts(pts1, pts2)
