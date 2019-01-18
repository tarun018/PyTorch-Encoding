###########################################################################
# Created by: Houjing Huang
# Copyright (c) 2018
###########################################################################

import os
import numpy as np
import random
from PIL import Image, ImageOps, ImageFilter

import torch
from .base import BaseDataset


class COCOPart(BaseDataset):
    BASE_DIR = 'coco_part'
    NUM_CLASS = 8

    def __init__(self, root='dataset', split='train', mode=None, transform=None, target_transform=None, base_size=192, crop_size=256):
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"
        self.images, self.masks = _get_im_mask_paths(os.path.join(root, split + '.txt'), root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.blur = False
        self.multi_scale_train = True
        self.scale_low = 0.75
        self.scale_high = 1.25
        self.im_pad_v = 0
        self.mask_pad_v = 0

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        # Single-crop fast validation
        elif self.mode == 'fast_val':
            img, mask = self._val_sync_transform(img, mask)
        # Multi-scale multi-crop validation
        elif self.mode == 'ms_val':
            mask = self._mask_transform(mask)
        else:
            raise ValueError('Invalid mode {}'.format(self.mode))
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    def _val_sync_transform(self, img, mask):
        short_size = self.base_size
        crop_size = self.crop_size
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            pad_left = random.randint(0, padw)
            pad_right = padw - pad_left
            pad_top = random.randint(0, padh)
            pad_bottom = padh - pad_top
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.im_pad_v)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.mask_pad_v)
        # center crop
        w, h = img.size
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if self.blur and (random.random() < 0.5):
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.choice([2, 3, 4, 5, 6])))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        if self.multi_scale_train:
            short_size = random.randint(int(self.base_size * self.scale_low), int(self.base_size * self.scale_high))
        else:
            short_size = self.base_size
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            pad_left = random.randint(0, padw)
            pad_right = padw - pad_left
            pad_top = random.randint(0, padh)
            pad_bottom = padh - pad_top
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.im_pad_v)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.mask_pad_v)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if self.blur and (random.random() < 0.5):
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.choice([2, 3, 4, 5, 6])))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def _get_im_mask_paths(split_f, root):
    im_paths = []
    mask_paths = []
    with open(split_f, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for line in lines:
        im_path, mask_path = line.split('\t')
        im_path = os.path.join(root, im_path)
        mask_path = os.path.join(root, mask_path)
        assert os.path.isfile(im_path), "File {} does not exist!".format(im_path)
        assert os.path.isfile(mask_path), "File {} does not exist!".format(mask_path)
        im_paths.append(im_path)
        mask_paths.append(mask_path)
    if len(im_paths) == 0:
        raise RuntimeError('Found 0 images in {}!'.format(split_f))
    return im_paths, mask_paths
