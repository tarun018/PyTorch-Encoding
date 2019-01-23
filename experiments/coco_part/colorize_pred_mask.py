###########################################################################
# Created by: Houjing Huang
# Copyright (c) 2019
###########################################################################
from __future__ import print_function
import os
import sys
sys.path.insert(0, '.')
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse

from encoding.utils.file_utils import walkdir
from encoding.utils.vis_utils import mask_to_color_im
from encoding.utils.vis_utils import save_im


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_dir', type=str, default='exp/prediction', help='Contains predicted mask')
    parser.add_argument('--new_dir', type=str, default='exp/prediction_color', help='Contains colorized mask')
    parser.add_argument('--max_num', type=int, default=None, help='Max number of images to colorize')
    args = parser.parse_args()
    return args


def main(args):
    sub_paths = list(walkdir(args.ori_dir, exts=['.png'], sub_path=True))
    if args.max_num is not None:
        sub_paths = sub_paths[:args.max_num]
    ori_paths = [os.path.join(args.ori_dir, p) for p in sub_paths]
    new_paths = [os.path.join(args.new_dir, p) for p in sub_paths]
    for ori_p, new_p in tqdm(zip(ori_paths, new_paths), miniters=5, desc='Colorize', unit=' images'):
        ori_im = np.asarray(Image.open(ori_p))
        new_im = mask_to_color_im(ori_im)
        save_im(new_im, new_p)


if __name__ == "__main__":
    main(parse_args())
