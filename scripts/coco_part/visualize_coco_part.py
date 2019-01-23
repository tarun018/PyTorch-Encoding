from encoding.utils.json import load_json
from encoding.utils.dense_pose import GetDensePoseMask
from encoding.utils.vis import read_im
from encoding.utils.vis import save_im
import cv2
import numpy as np
import os.path as osp


def vis_ann(im_dir, ann):
    # All keys:
    #   'category_id', 'id', 'image_id',
    #   'iscrowd', 'area',
    #   'num_keypoints', 'keypoints',
    #   'bbox', 'segmentation', 'dp_masks',
    #   'dp_x', 'dp_y', 'dp_I', 'dp_U', 'dp_V'
    # print(ann.keys())
    part_mask = GetDensePoseMask(ann['dp_masks'])
    bbr = np.array(ann['bbox']).astype(int)  # the box.
    ################
    im = read_im(osp.join(im_dir, im_id_to_name[ann['image_id']]), resize_h_w=None, transpose=False)
    x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
    x2 = min([x2, im.shape[1]])
    y2 = min([y2, im.shape[0]])
    ################
    MaskIm = cv2.resize(part_mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
    MaskBool = np.tile((MaskIm == 0)[:, :, np.newaxis], [1, 1, 3])
    #  Replace the visualized mask image with I_vis.
    Mask_vis = cv2.applyColorMap((MaskIm * 15).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
    Mask_vis[MaskBool] = im[y1:y2, x1:x2, :][MaskBool]
    im = im.copy().astype(np.float)
    im[y1:y2, x1:x2, :] = im[y1:y2, x1:x2, :] * 0.3 + Mask_vis * 0.7
    return im


im_dir = '/home/users/houjing.huang/Dataset/coco/val2014/'
ann_json = '/home/users/houjing.huang/Dataset/coco/annotations/densepose_coco_2014_minival.json'
ann = load_json(ann_json)
im_id_to_name = {rec['id']: rec['file_name'] for rec in ann['images']}
save_dir = 'exp/vis_dense_pose/val_set'

i = 0
for ann_ in ann['annotations']:
    if not 'dp_masks' in ann_:
        continue
    i += 1
    if i == 11:
        break
    im = vis_ann(im_dir, ann_)
    save_name = im_id_to_name[ann_['image_id']][:-4] + '_' + str(ann_['id']) + '.jpg'
    save_im(im, osp.join(save_dir, save_name), transpose=False, check_bound=True)