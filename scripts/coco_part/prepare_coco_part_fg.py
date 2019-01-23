"""
cd ${DANet_DIR}
python scripts/prepare_coco_part_fg.py
"""
import os.path as osp
import numpy as np
import cv2
from encoding.utils.json import load_json
from encoding.utils.dense_pose import GetDensePoseMask
from encoding.utils.vis import read_im
from encoding.utils.vis import save_im
from encoding.utils.vis import get_im_names
from encoding.utils.utils import may_make_dir

import sys
sys.path.insert(0, '../cocoapi/PythonAPI')
from pycocotools import mask as maskUtils


def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annToMask(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, h, w)
    m = maskUtils.decode(rle)
    return m


def get_im_id_to_name(ann):
    """input ann: loaded json"""
    return {rec['id']: rec['file_name'] for rec in ann['images']}


def transform(ann, im_dir, save_im_dir, save_mask_dir, split_file, to_save_im=False):
    """
    Args:
        ann: loaded Densepose annotation json
        to_save_im: If your cropped images have been saved when preparing part parsing, this can be False
    """
    may_make_dir(osp.dirname(split_file))
    fid = open(split_file, 'w')
    im_id_to_name = get_im_id_to_name(ann)
    anns = ann['annotations']
    i = 0
    n_dp = 0
    n_fg = 0
    for ann in anns:
        i += 1
        if not 'dp_masks' in ann:
            continue
        n_dp += 1
        if not 'segmentation' in ann:
            continue
        n_fg += 1
        bbox = np.array(ann['bbox']).astype(int)
        im = read_im(osp.join(im_dir, im_id_to_name[ann['image_id']]), resize_h_w=None, transpose=False)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        x1 = max([x1, 0])
        y1 = max([y1, 0])
        x2 = min([x2, im.shape[1]])
        y2 = min([y2, im.shape[0]])
        # The same resolution as whole image.
        MaskIm = annToMask(ann, im.shape[0], im.shape[1])
        assert len(MaskIm.shape) == 2, "len(MaskIm.shape) {}".format(len(MaskIm.shape))
        assert MaskIm.shape == im.shape[:2], "MaskIm.shape {}, im.shape {}".format(MaskIm.shape, im.shape)
        # print('=> im.shape', im.shape)
        # print('=> MaskIm.shape', MaskIm.shape)
        # print('=> MaskIm.dtype', MaskIm.dtype)
        # print('=> np.unique(MaskIm)', np.unique(MaskIm))
        # The same resolution as bbox
        MaskIm = MaskIm[y1:y2, x1:x2]
        # print('=> MaskIm.shape', MaskIm.shape)
        check_mask_value_(MaskIm)
        save_im_name = im_id_to_name[ann['image_id']][:-4] + '_' + str(ann['id']) + '.jpg'
        save_im_path = osp.join(save_im_dir, save_im_name)
        if to_save_im:
            im = im[y1:y2, x1:x2, :]
            save_im(im, save_im_path, transpose=False, check_bound=False)
        save_mask_name = im_id_to_name[ann['image_id']][:-4] + '_' + str(ann['id']) + '_fg' + '.png'
        save_mask_path = osp.join(save_mask_dir, save_mask_name)
        save_im(MaskIm, save_mask_path, transpose=False, check_bound=False)
        rel_im_path = '/'.join(save_im_path.split('/')[-3:])
        rel_mask_path = '/'.join(save_mask_path.split('/')[-3:])
        fid.write('{}\t{}\n'.format(rel_im_path, rel_mask_path))
        if i % 200 == 0:
            print('{}/{} Done'.format(i, len(anns)))
    fid.close()
    return len(anns), n_dp, n_fg

###################

def print_num_ims(sub_dir, split, pattern):
    im_dir = osp.join(save_root_dir, sub_dir, split)
    im_names = get_im_names(im_dir, pattern=pattern, return_np=False, return_path=True)
    print('{} {} images in {}, {}'.format(len(im_names), split, im_dir, pattern))


def check_mask_value_(mask):
    unique_mask_v = np.unique(mask)
    assert np.array_equal(unique_mask_v, np.array([0])) or np.array_equal(unique_mask_v, np.array([1])) or np.array_equal(unique_mask_v, np.array([0, 1])), "unique_mask_v is {}".format(unique_mask_v)
    

def check_mask_value(split):
    """To make sure saving and loading mask images are correct."""
    mask_files = get_im_names(osp.join(save_root_dir, mask_dir, split), pattern='*.png', return_np=False, return_path=True)
    for mf in mask_files:
        mask = read_im(mf, convert_rgb=False, resize_h_w=None, transpose=False)
        check_mask_value_(mask)


def vis_mask_(im, mask, save_path):
    assert im.shape[:2] == mask.shape
    ori_mask = mask
    mask = mask[..., np.newaxis] * np.array([[[255, 0, 0]]])
    mask[ori_mask == 0] = im[ori_mask == 0]
    im = im * 0.3 + mask * 0.7
    save_im(im, save_path, transpose=False, check_bound=True)


def vis_mask(split, num=10, save_dir=''):
    im_files = get_im_names(osp.join(save_root_dir, 'images', split), pattern='*.jpg', return_np=False, return_path=True)
    mask_files = get_im_names(osp.join(save_root_dir, mask_dir, split), pattern='*.png', return_np=False, return_path=True)
    im_files.sort()
    mask_files.sort()
    assert len(im_files) == len(mask_files)
    assert all([osp.basename(im_f)[:-4] + '_fg' == osp.basename(mask_f)[:-4] for im_f, mask_f in zip(im_files, mask_files)])
    for im_f, mask_f in list(zip(im_files, mask_files))[:num]:
        vis_mask_(
            read_im(im_f, convert_rgb=True, resize_h_w=None, transpose=False),
            read_im(mask_f, convert_rgb=False, resize_h_w=None, transpose=False),
            osp.join(save_dir, osp.basename(im_f))
        )

#############

ann_dir = '/mnt/data-1/data/houjing.huang/Dataset/coco/annotations'
im_root_dir = '/mnt/data-1/data/houjing.huang/Dataset/coco'
save_root_dir = 'datasets/coco_part'
mask_dir = 'masks_fg'

# After saving files, you can switch this on to make sure no error.
only_making_sure_no_eror = False

if not only_making_sure_no_eror:
    n_ann, n_dp_ann, n_fg_ann = transform(
        load_json(osp.join(ann_dir, 'densepose_coco_2014_train.json')),
        osp.join(im_root_dir, 'train2014'),
        osp.join(save_root_dir, 'images', 'train'),
        osp.join(save_root_dir, mask_dir, 'train'),
        osp.join(save_root_dir, 'train_fg.txt')
    )
    print('densepose_coco_2014_train.json, {} annotations, {} with dense_pose, {} with dense_pose+foreground'.format(n_ann, n_dp_ann, n_fg_ann))

    n_ann, n_dp_ann, n_fg_ann = transform(
        load_json(osp.join(ann_dir, 'densepose_coco_2014_valminusminival.json')),
        osp.join(im_root_dir, 'val2014'),
        osp.join(save_root_dir, 'images', 'train'),
        osp.join(save_root_dir, mask_dir, 'train'),
        osp.join(save_root_dir, 'train_fg.txt')
    )
    print('densepose_coco_2014_valminusminival.json, {} annotations, {} with dense_pose, {} with dense_pose+foreground'.format(n_ann, n_dp_ann, n_fg_ann))

    n_ann, n_dp_ann, n_fg_ann = transform(
        load_json(osp.join(ann_dir, 'densepose_coco_2014_minival.json')),
        osp.join(im_root_dir, 'val2014'),
        osp.join(save_root_dir, 'images', 'val'),
        osp.join(save_root_dir, mask_dir, 'val'),
        osp.join(save_root_dir, 'val_fg.txt')
    )
    print('densepose_coco_2014_minival.json, {} annotations, {} with dense_pose, {} with dense_pose+foreground'.format(n_ann, n_dp_ann, n_fg_ann))

######################

print('Making Sure No Error...')

print_num_ims('images', 'train', '*.jpg')
print_num_ims('images', 'val', '*.jpg')
print_num_ims(mask_dir, 'train', '*.png')
print_num_ims(mask_dir, 'val', '*.png')

check_mask_value('train')
check_mask_value('val')

vis_mask('train', save_dir=osp.join('exp/vis_coco_part_fg', 'train'))
vis_mask('val', save_dir=osp.join('exp/vis_coco_part_fg', 'val'))

print('Done, Seems No Error!')