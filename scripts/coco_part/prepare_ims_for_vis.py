"""Copy some images from different datasets to a directory, for visualizing part segmentation."""
import sys
sys.path.insert(0, '.')

import numpy as np
import os
from encoding.utils.file_utils import walkdir
from encoding.utils.file_utils import copy_to


def get_random_im_paths(directory, num):
    im_paths = list(walkdir(directory, exts=['.jpg', '.png']))
    im_paths = sorted(im_paths)
    np.random.RandomState(seed=1).shuffle(im_paths)
    assert len(im_paths) >= num, "Not enough images, {} < {}".format(len(im_paths), num)
    return im_paths[:num]


im_dirs = [
    '/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet_Data/dataset/market1501/Market-1501-v15.09.15/bounding_box_train',
    '/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet_Data/dataset/cuhk03_np_detected_jpg/cuhk03-np/detected/bounding_box_train',
    '/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet_Data/dataset/duke/DukeMTMC-reID/bounding_box_train',
    '/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet_Data/dataset/msmt17/MSMT17_V1/train',
    '/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet_Data/dataset/partial_reid/Partial-REID_Dataset',
    '/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet_Data/dataset/partial_ilids/Partial_iLIDS',
    '/mnt/data-1/data/houjing.huang/Project/DANet_cluster/upload_folder/DANet/datasets/yating.liu/crop_bbx/gt',
]
dest_dir = '/mnt/data-1/data/houjing.huang/Project/PyTorch-Encoding/dataset/reid_ims_to_vis'

im_paths = []
for im_dir, ext in im_dirs:
    im_paths += get_random_im_paths(im_dir, 16)

for im_p in im_paths:
    copy_to(im_p, os.path.join(dest_dir, '_'.join(im_p.split('/')[-3:])))
