"""
Annotation: this file split train and val samples from instance_npz into instance_dataset.
Anthor: Lin, PRML Lab
Data: 20, July, 2022
"""

import os
import shutil
from tqdm import tqdm
# from random import randint
from config import config

ratio = 0.8  # for train

instance_npz = config.instance_npz
instance_train = config.instance_train
instance_val = config.instance_val

if os.path.exists(instance_train):
    shutil.rmtree(instance_train)
    os.mkdir(instance_train)

if os.path.exists(instance_val):
    shutil.rmtree(instance_val)
    os.mkdir(instance_val)

train_list = []
test_list = []

npz_list = os.listdir(instance_npz)
train_num = int(ratio * len(npz_list))

for npz_name in tqdm(npz_list[:train_num]):
    npz_name_full = os.path.join(instance_npz, npz_name)
    shutil.copy(npz_name_full, instance_train)

for npz_name in tqdm(npz_list[train_num:]):
    npz_name_full = os.path.join(instance_npz, npz_name)
    shutil.copy(npz_name_full, instance_val)

