"""
cd ${DANet_DIR}
python scripts/prepare_coco_part.py
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


def get_im_id_to_name(ann):
    """input ann: loaded json"""
    return {rec['id']: rec['file_name'] for rec in ann['images']}


def transform(ann, im_dir, save_im_dir, save_mask_dir, split_file):
    """input ann: loaded json"""
    may_make_dir(osp.dirname(split_file))
    fid = open(split_file, 'w')
    im_id_to_name = get_im_id_to_name(ann)
    anns = ann['annotations']
    i = 0
    for ann in anns:
        if not 'dp_masks' in ann:
            continue
        bbox = np.array(ann['bbox']).astype(int)
        im = read_im(osp.join(im_dir, im_id_to_name[ann['image_id']]), resize_h_w=None, transpose=False)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        x1 = max([x1, 0])
        y1 = max([y1, 0])
        x2 = min([x2, im.shape[1]])
        y2 = min([y2, im.shape[0]])
        mask = GetDensePoseMask(ann['dp_masks']).astype(np.uint8)
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        assert mask.min() >= 0, "mask.min is {}".format(mask.min())
        assert mask.max() <= 14, "mask.max is {}".format(mask.max())
        im = im[y1:y2, x1:x2, :]
        save_im_name = im_id_to_name[ann['image_id']][:-4] + '_' + str(ann['id']) + '.jpg'
        save_im_path = osp.join(save_im_dir, save_im_name)
        save_im(im, save_im_path, transpose=False, check_bound=False)
        save_mask_name = im_id_to_name[ann['image_id']][:-4] + '_' + str(ann['id']) + '.png'
        save_mask_path = osp.join(save_mask_dir, save_mask_name)
        save_im(mask, save_mask_path, transpose=False, check_bound=False)
        rel_im_path = '/'.join(save_im_path.split('/')[-3:])
        rel_mask_path = '/'.join(save_mask_path.split('/')[-3:])
        fid.write('{}\t{}\n'.format(rel_im_path, rel_mask_path))
        i += 1
    fid.close()
    return len(anns), i


ann_dir = '/mnt/data-1/data/houjing.huang/Dataset/coco/annotations'
im_root_dir = '/mnt/data-1/data/houjing.huang/Dataset/coco'
save_root_dir = 'datasets/coco_part'
mask_dir = 'masks_14_parts'

only_making_sure_no_eror = True

if not only_making_sure_no_eror:
    n_ann, n_dp_ann = transform(
        load_json(osp.join(ann_dir, 'densepose_coco_2014_train.json')),
        osp.join(im_root_dir, 'train2014'),
        osp.join(save_root_dir, 'images', 'train'),
        osp.join(save_root_dir, mask_dir, 'train'),
        osp.join(save_root_dir, 'train.txt')
    )
    print('densepose_coco_2014_train.json, {} annotations, {} with dense pose'.format(n_ann, n_dp_ann))

    n_ann, n_dp_ann = transform(
        load_json(osp.join(ann_dir, 'densepose_coco_2014_valminusminival.json')),
        osp.join(im_root_dir, 'val2014'),
        osp.join(save_root_dir, 'images', 'train'),
        osp.join(save_root_dir, mask_dir, 'train'),
        osp.join(save_root_dir, 'train.txt')
    )
    print('densepose_coco_2014_valminusminival.json, {} annotations, {} with dense pose'.format(n_ann, n_dp_ann))

    n_ann, n_dp_ann = transform(
        load_json(osp.join(ann_dir, 'densepose_coco_2014_minival.json')),
        osp.join(im_root_dir, 'val2014'),
        osp.join(save_root_dir, 'images', 'val'),
        osp.join(save_root_dir, mask_dir, 'val'),
        osp.join(save_root_dir, 'val.txt')
    )
    print('densepose_coco_2014_minival.json, {} annotations, {} with dense pose'.format(n_ann, n_dp_ann))

######################

print('Making Sure No Error...')


def print_num_ims(sub_dir, split, pattern):
    im_dir = osp.join(save_root_dir, sub_dir, split)
    im_names = get_im_names(im_dir, pattern=pattern, return_np=False, return_path=True)
    print('{} {} images in {}, {}'.format(len(im_names), split, im_dir, pattern))


def check_mask_value_(mask_file):
    mask = read_im(mask_file, convert_rgb=False, resize_h_w=None, transpose=False)
    assert len(mask.shape) == 2, "{} mask.shape is {}".format(mask_file, mask.shape)
    assert mask.min() >= 0, "{} mask.min is {}".format(mask_file, mask.min())
    assert mask.max() <= 14, "{} mask.max is {}".format(mask_file, mask.max())


def check_mask_value(split):
    mask_files = get_im_names(osp.join(save_root_dir, mask_dir, split), pattern='*.png', return_np=False, return_path=True)
    for mf in mask_files:
        check_mask_value_(mf)


def vis_mask_(im, mask, save_path):
    assert im.shape[:2] == mask.shape
    mask = cv2.applyColorMap((mask * 15).astype(np.uint8), cv2.COLORMAP_PARULA)
    im = im * 0.3 + mask * 0.7
    save_im(im, save_path, transpose=False, check_bound=True)


def vis_mask(split, num=10, save_dir=''):
    im_files = get_im_names(osp.join(save_root_dir, 'images', split), pattern='*.jpg', return_np=False, return_path=True)
    mask_files = get_im_names(osp.join(save_root_dir, mask_dir, split), pattern='*.png', return_np=False, return_path=True)
    im_files.sort()
    mask_files.sort()
    assert len(im_files) == len(mask_files)
    assert all([osp.basename(im_f)[:-4] == osp.basename(mask_f)[:-4] for im_f, mask_f in zip(im_files, mask_files)])
    for im_f, mask_f in list(zip(im_files, mask_files))[:num]:
        vis_mask_(
            read_im(im_f, convert_rgb=True, resize_h_w=None, transpose=False),
            read_im(mask_f, convert_rgb=False, resize_h_w=None, transpose=False),
            osp.join(save_dir, osp.basename(im_f))
        )


print_num_ims('images', 'train', '*.jpg')
print_num_ims('images', 'val', '*.jpg')
print_num_ims(mask_dir, 'train', '*.png')
print_num_ims(mask_dir, 'val', '*.png')

check_mask_value('train')
check_mask_value('val')

vis_mask('train', save_dir=osp.join('exp/vis_coco_part', 'train'))
vis_mask('val', save_dir=osp.join('exp/vis_coco_part', 'val'))

print('Done, Seems No Error!')