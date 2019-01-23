"""This file should be run in ${DANet} dir"""
import cv2
import numpy as np
from encoding.utils.vis import make_im_grid, save_im
from encoding.utils.vis import read_im
from encoding.utils.utils import read_lines


def may_resize_im(im, resize_h_w=None, interpolation=cv2.INTER_LINEAR):
    if resize_h_w is not None and (im.shape[0], im.shape[1]) != resize_h_w:
        im = cv2.resize(im, resize_h_w[::-1], interpolation=interpolation)
    return im


def mask_to_im(mask, nclass, transpose=False):
    """input mask: shape [H, W]
    return mask: shape [3, H, W] if transpose=True, else [H, W, 3]
    """
    mask = cv2.applyColorMap((mask * (255 // (nclass - 1))).astype(np.uint8), cv2.COLORMAP_PARULA)
    if transpose:
        mask = mask.transpose(2, 0, 1)
    return mask


def vis_one_im(im, mask, nclass, resize_h_w=(128, 64), im_grid_list=None):
    """
    im: shape [H, W, 3]
    mask: shape [h, w]
    """
    assert len(im.shape) == 3
    assert im.shape[2] == 3
    assert len(mask.shape) == 2
    im = may_resize_im(im, resize_h_w=resize_h_w).transpose(2, 0, 1)
    mask = may_resize_im(mask, resize_h_w=resize_h_w, interpolation=cv2.INTER_NEAREST)
    mask = mask_to_im(mask, nclass, transpose=True)
    if im_grid_list is None:
        im_grid_list = []
    im_grid_list.extend([im, mask])
    return im_grid_list

# Including Background class
nclass = 8
# infer_im_path_mask_path_file = 'datasets/cuhk03/detected/infer_im_path_mask_path.txt'
infer_im_path_mask_path_file = 'datasets/duke/infer_im_path_mask_path.txt'
lines = read_lines(infer_im_path_mask_path_file, strip=True)
im_paths, mask_paths = zip(*[l.split('\t') for l in lines])
rand_inds = np.random.permutation(range(len(im_paths)))
im_paths = np.array(im_paths)[rand_inds]
mask_paths = np.array(mask_paths)[rand_inds]
num_vis_ims = 64
im_grid_list = []
for i in range(num_vis_ims):
    im = read_im(im_paths[i], convert_rgb=True, resize_h_w=None, transpose=False)
    mask = read_im(mask_paths[i], convert_rgb=False, resize_h_w=None, transpose=False)
    vis_one_im(im, mask, nclass, resize_h_w=(128, 64), im_grid_list=im_grid_list)
n_cols = 8
n_rows = int(np.ceil(len(im_grid_list) / n_cols))
vis_im = make_im_grid(im_grid_list, n_rows, n_cols, 8, 255)
save_im(vis_im, 'exp/vis_coco_part_pred/cuhk03_detected/vis_im.jpg', transpose=True, check_bound=True)
