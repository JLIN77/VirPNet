import os


class Config(object):

    model = 'ResNet34'
    kitti_path = './dataset_split/KITTI'
    image_path = kitti_path + '/object/training/image_2'
    instance_npz = './instance_npz'
    instance_pts_npz = './instance_pts_npz'
    label_path = './instance_label'
    recorder_txt = './record/record3.txt'

    instance_train = './instance_dataset/train'
    instance_val = './instance_dataset/val'

    lr_init = 0.005    # 0.1 seems too big, 0.01 seems little big
    lr_list = [10, 20]

    batch_size = 3
    num_epoch = 30
    save_interval = 5

    # for inference
    # checkpoint = './checkpoints/epoch_10.pth'
    checkpoint = '../virtual_pointnet2/checkpoints15/epoch_50.pth'
    batch_size_test = 1

    MAX_DISTANCE = 70.4
    MAX_DISTANCE_X = 70.4
    MAX_DISTANCE_Y = 40.0
    MAX_DISTANCE_Z = 3.0

    GRID_SIZE = 10

    inference_res = './inference_res'

    POINT_CLOUD_RANGE = [0, -40, -3, 70.4, 40, 1]   # X_MIM, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX

    resume = False
    VP_NUM = 2
    GRID_POINT_NUM = 10

    box_padding = 4


config = Config()



