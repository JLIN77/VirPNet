"""
Annotation: this file split pedestrian/car/bicycle instances from original KITTI 3D Detection dataset into dataset_split.
Anthor: Lin, PRML Lab
Data: 20, July, 2022
"""

import os

import cv2
from tqdm import tqdm
import shutil

kitti_path = '/home/car/Datasets/KITTI'
instance = 'Pedestrian'   # 'Pedestrian':1779,   Cyclist: 1141    (1779, 1141)=2486

save_path = './dataset_split/KITTI/object/testing'

def check_some(im_idx):
    im_name_full = kitti_path + '/object/testing/image_2/' + im_idx + '.png'
    im = cv2.imread(im_name_full)
    cv2.imshow(str(im_idx), im)
    cv2.waitKey(1000)


if __name__ == '__main__':
    folders = ['label_2', 'image_2', 'velodyne', 'calib']
    label_path = kitti_path + '/object/testing/label_2'
    image_path = kitti_path + '/object/testing/image_2'
    velo_path = kitti_path + '/object/testing/velodyne'
    calib_path = kitti_path + '/object/testing/calib'
    # plane_path = kitti_path + '/object/testing/planes'

    instance_list = []
    for label_file in tqdm(os.listdir(label_path)):
        # if label_file != '003862.txt':
        #     continue
        label_file_full = os.path.join(label_path, label_file)
        with open(label_file_full) as f:
            rds = f.readlines()
            for rd in rds:
                info = rd.strip().split(' ')
                if info[0] == instance and float(info[-1]) > 0.5:
                    instance_list.append(label_file[:-4])
                    print('{}: box_2d {}, box_3d {}'.format(label_file, info[4:8], info[8:11]))
                    for folder in folders:
                        if folder == folders[0]:
                            src = os.path.join(label_path, label_file[:-4] + '.txt')
                            dst = os.path.join(save_path, folder)
                            shutil.copy(src, dst)
                        elif folder == folders[1]:
                            src = os.path.join(image_path, label_file[:-4] + '.png')
                            dst = os.path.join(save_path, folder)
                            shutil.copy(src, dst)
                        elif folder == folders[2]:
                            src = os.path.join(velo_path, label_file[:-4] + '.bin')
                            dst = os.path.join(save_path, folder)
                            shutil.copy(src, dst)
                        elif folder == folders[3]:
                            src = os.path.join(calib_path, label_file[:-4] + '.txt')
                            dst = os.path.join(save_path, folder)
                            shutil.copy(src, dst)
                        # elif folder == folders[4]:
                        #     src = os.path.join(plane_path, label_file[:-4] + '.txt')
                        #     dst = os.path.join(save_path, folder)
                        #     shutil.copy(src, dst)
                        else:
                            print('sth wrong')
                    break
                else:
                    continue
    # print(category_list)         # 3155
    print(len(instance_list))

    print(instance_list[:5])
    for im_idx in instance_list[:5]:
        check_some(im_idx)

