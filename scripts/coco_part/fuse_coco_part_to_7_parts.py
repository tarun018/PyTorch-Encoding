"""
cd ${DANet_DIR}
python scripts/fuse_coco_part_to_7_parts.py
"""
import os.path as osp
import numpy as np
import cv2
from encoding.utils.vis import read_im
from encoding.utils.vis import save_im
from encoding.utils.vis import get_im_names
from encoding.utils.vis import make_im_grid

"""Original
0: background
1: torso
2: right hand
3: left hand
4: left foot
5: right foot
6: right upper leg
7: left upper leg
8: right lower leg
9: left lower leg
10: left upper arm
11: right upper arm
12: left lower arm
13: right lower arm
14: head
"""

"""Fused To 7 Parts
0: background
1: head
2: torso
3: upper arm
4: lower arm (including hand)
5: upper leg
6: lower leg
7: foot
"""

mapping = [
    (0, 0),
    (1, 2),
    (2, 4),
    (3, 4),
    (4, 7),
    (5, 7),
    (6, 5),
    (7, 5),
    (8, 6),
    (9, 6),
    (10, 3),
    (11, 3),
    (12, 4),
    (13, 4),
    (14, 1),
]
root_dir = 'datasets/coco_part'
ori_mask_dir = 'masks_14_parts'
new_mask_dir = 'masks_7_parts'


def vis_mask_each_part_no_superimpose_(im, mask, save_path, num_parts):
    """
    im: [H, W, 3]
    mask: [H, W]
    """
    assert im.shape[:2] == mask.shape
    ims = [im.transpose(2, 0, 1)]
    for i in range(num_parts + 1):
        m = np.zeros_like(mask)
        m[mask == i] = 255
        m = np.repeat(m[np.newaxis, ...], 3, 0)
        ims.append(m)
    im = make_im_grid(ims, 1, len(ims), 8, 128)
    save_im(im, save_path, transpose=True, check_bound=True)


def vis_mask_each_part_(im, mask, save_path, num_parts):
    """
    im: [H, W, 3]
    mask: [H, W]
    """
    assert im.shape[:2] == mask.shape
    H, W = mask.shape
    # [3, H, W]
    im = im.transpose(2, 0, 1)
    # dim the image
    dim_im = im * 0.5
    ims = [im]
    for i in range(num_parts + 1):
        # [3, H, W]
        m = np.array([0, 255, 0])[:, np.newaxis, np.newaxis].repeat(H, 1).repeat(W, 2)
        dim_im_copy = dim_im.copy()
        dim_im_copy[:, mask == i] = dim_im_copy[:, mask == i] * 0.4 + m[:, mask == i] * 0.6
        ims.append(dim_im_copy)
    im = make_im_grid(ims, 1, len(ims), 8, 128)
    save_im(im, save_path, transpose=True, check_bound=True)


# TOTALLY MISTAKE!
# def fuse_parts_(mask_f, new_mask_f):
#     # If not copied, it throws an error `ValueError: assignment destination is read-only`. Confused.
#     mask = read_im(mask_f, convert_rgb=False, resize_h_w=None, transpose=False).copy()
#     for m in mapping:
#         mask[mask == m[0]] = m[1]
#     save_im(mask, new_mask_f, transpose=False, check_bound=False)


def fuse_parts_(mask_f, new_mask_f):
    ori_mask = read_im(mask_f, convert_rgb=False, resize_h_w=None, transpose=False)
    mask = ori_mask.copy()
    for m in mapping:
        mask[ori_mask == m[0]] = m[1]
    save_im(mask, new_mask_f, transpose=False, check_bound=False)


def fuse_parts(split):
    mask_files = get_im_names(osp.join(root_dir, ori_mask_dir, split), pattern='*.png', return_np=False, return_path=True)
    mask_files.sort()
    for mask_f in mask_files:
        fuse_parts_(mask_f, osp.join(root_dir, new_mask_dir, split, osp.basename(mask_f)))


only_making_sure_no_eror = True
if not only_making_sure_no_eror:
    fuse_parts('train')
    fuse_parts('val')

######################

print('Making Sure No Error...')


def print_num_ims(sub_dir, split, pattern):
    im_dir = osp.join(root_dir, sub_dir, split)
    im_names = get_im_names(im_dir, pattern=pattern, return_np=False, return_path=True)
    print('{} {} images in {}, {}'.format(len(im_names), split, im_dir, pattern))


def check_mask_value_(mask_file, max_v):
    mask = read_im(mask_file, convert_rgb=False, resize_h_w=None, transpose=False)
    assert len(mask.shape) == 2, "{} mask.shape is {}".format(mask_file, mask.shape)
    assert mask.min() >= 0, "{} mask.min is {}".format(mask_file, mask.min())
    assert mask.max() <= max_v, "{} mask.max is {}".format(mask_file, mask.max())


def check_mask_value(sub_dir, max_v, split):
    mask_files = get_im_names(osp.join(root_dir, sub_dir, split), pattern='*.png', return_np=False, return_path=True)
    for mf in mask_files:
        check_mask_value_(mf, max_v)


def vis_mask_(im, mask, save_path, num_parts):
    assert im.shape[:2] == mask.shape
    mask = cv2.applyColorMap((mask * (255 // num_parts)).astype(np.uint8), cv2.COLORMAP_PARULA)
    im = im * 0.2 + mask * 0.8
    save_im(im, save_path, transpose=False, check_bound=True)


def vis_mask(sub_dir, split, num=10, save_dir='', num_parts=14):
    im_files = get_im_names(osp.join(root_dir, 'images', split), pattern='*.jpg', return_np=False, return_path=True)
    mask_files = get_im_names(osp.join(root_dir, sub_dir, split), pattern='*.png', return_np=False, return_path=True)
    im_files.sort()
    mask_files.sort()
    assert len(im_files) == len(mask_files)
    assert all([osp.basename(im_f)[:-4] == osp.basename(mask_f)[:-4] for im_f, mask_f in zip(im_files, mask_files)])
    for im_f, mask_f in list(zip(im_files, mask_files))[:num]:
        for vis_func, save_sub_dir in zip([vis_mask_, vis_mask_each_part_], ['all_on_one_im', 'one_im_per_mask']):
            vis_func(
                read_im(im_f, convert_rgb=True, resize_h_w=None, transpose=False),
                read_im(mask_f, convert_rgb=False, resize_h_w=None, transpose=False),
                osp.join(save_dir, save_sub_dir, osp.basename(im_f)),
                num_parts
            )


print_num_ims('images', 'train', '*.jpg')
print_num_ims('images', 'val', '*.jpg')
print_num_ims(ori_mask_dir, 'train', '*.png')
print_num_ims(ori_mask_dir, 'val', '*.png')
print_num_ims(new_mask_dir, 'train', '*.png')
print_num_ims(new_mask_dir, 'val', '*.png')

check_mask_value(ori_mask_dir, 14, 'train')
check_mask_value(ori_mask_dir, 14, 'val')
check_mask_value(new_mask_dir, 7, 'train')
check_mask_value(new_mask_dir, 7, 'val')

vis_mask(ori_mask_dir, 'train', save_dir=osp.join('exp/vis_coco_part_7_parts/14_parts', 'train'), num_parts=14)
vis_mask(ori_mask_dir, 'val', save_dir=osp.join('exp/vis_coco_part_7_parts/14_parts', 'val'), num_parts=14)
vis_mask(new_mask_dir, 'train', save_dir=osp.join('exp/vis_coco_part_7_parts/7_parts', 'train'), num_parts=7)
vis_mask(new_mask_dir, 'val', save_dir=osp.join('exp/vis_coco_part_7_parts/7_parts', 'val'), num_parts=7)

print('Done, Seems No Error!')