"""Save the im name to keypoint mapping from the COCO Densepose dataset."""
import sys
sys.path.insert(0, '.')
import os.path as osp
import numpy as np
import cv2
from package.utils.utils import load_json
from package.utils.visualization import read_im
from package.utils.visualization import save_im
from package.utils.dataset_utils import get_im_names
from package.utils.utils import save_pickle


def get_im_id_to_name(ann):
    """input ann: loaded json"""
    return {rec['id']: rec['file_name'] for rec in ann['images']}


#TODO: visualize keypoints on image
def transform(ann, im_dir):
    """input ann: loaded json"""
    # area, bbox, category_id, dp_I, dp_U, dp_V, dp_masks, dp_x, dp_y, id, image_id, iscrowd, keypoints, num_keypoints, segmentation
    # print(', '.join(ann['annotations'][0].keys()))
    im_id_to_name = get_im_id_to_name(ann)
    anns = ann['annotations']
    i = 0
    n_dp = 0
    n_kpt = 0
    im_name_to_kpt = {}
    im_name_to_h_w = {}
    for ann in anns:
        if not 'dp_masks' in ann:
            continue
        n_dp += 1
        if not 'keypoints' in ann:
            continue
        n_kpt += 1

        bbox = np.array(ann['bbox']).astype(int)
        # im = read_im(osp.join(im_dir, im_id_to_name[ann['image_id']]), resize_h_w=None, transpose=False)
        w, h = bbox[2], bbox[3]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        x1 = max([x1, 0])
        y1 = max([y1, 0])
        # x2 = min([x2, im.shape[1]])
        # y2 = min([y2, im.shape[0]])

        # type(loaded['annotations'][0]['keypoints']) is list, length is 51
        kpt = np.array(ann['keypoints']).reshape([17, 3])
        kpt[:, 0] -= x1
        kpt[:, 1] -= y1
        kpt[:, 0].clip(0, w)
        kpt[:, 1].clip(0, h)
        unique_v = np.unique(kpt[:, 2])

        assert np.array_equal(unique_v, np.array([0])) \
               or np.array_equal(unique_v, np.array([1])) \
               or np.array_equal(unique_v, np.array([2])) \
               or np.array_equal(unique_v, np.array([0, 1])) \
               or np.array_equal(unique_v, np.array([0, 2])) \
               or np.array_equal(unique_v, np.array([1, 2])) \
               or np.array_equal(unique_v, np.array([0, 1, 2])), "unique_v is {}".format(unique_v)
        save_im_name = im_id_to_name[ann['image_id']][:-4] + '_' + str(ann['id']) + '.jpg'
        im_name_to_kpt[save_im_name] = kpt
        im_name_to_h_w[save_im_name] = (h, w)
        i += 1
    print('{} annotations, {} with dp_masks, {} with keypoints'.format(len(anns), n_dp, n_kpt))
    return im_name_to_kpt, im_name_to_h_w


im_name_to_kpt = {}
im_name_to_h_w = {}
ann_dir = '/mnt/data-1/data/houjing.huang/Dataset/coco/annotations'
im_root_dir = '/mnt/data-1/data/houjing.huang/Dataset/coco'
# 100403 annotations, 39210 with dp_masks, 39210 with keypoints
im_name_to_kpt_, im_name_to_h_w_ = transform(load_json(osp.join(ann_dir, 'densepose_coco_2014_train.json')), osp.join(im_root_dir, 'train2014'))
im_name_to_kpt.update(im_name_to_kpt_)
im_name_to_h_w.update(im_name_to_h_w_)
# 25053 annotations, 7297 with dp_masks, 7297 with keypoints
im_name_to_kpt_, im_name_to_h_w_ = transform(load_json(osp.join(ann_dir, 'densepose_coco_2014_valminusminival.json')), osp.join(im_root_dir, 'val2014'))
im_name_to_kpt.update(im_name_to_kpt_)
im_name_to_h_w.update(im_name_to_h_w_)
# 5673 annotations, 2243 with dp_masks, 2243 with keypoints
im_name_to_kpt_, im_name_to_h_w_ = transform(load_json(osp.join(ann_dir, 'densepose_coco_2014_minival.json')), osp.join(im_root_dir, 'val2014'))
im_name_to_kpt.update(im_name_to_kpt_)
im_name_to_h_w.update(im_name_to_h_w_)
# Done, 48750 im_name to kpt mappings
print('Done, {} im_name to kpt mappings'.format(len(im_name_to_kpt)))
save_pickle(im_name_to_kpt, '/mnt/data-1/data/houjing.huang/Dataset/coco_part/im_name_to_kpt.pkl')
save_pickle(im_name_to_h_w, '/mnt/data-1/data/houjing.huang/Dataset/coco_part/im_name_to_h_w.pkl')