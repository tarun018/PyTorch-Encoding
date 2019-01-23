###########################################################################
# Created by: Houjing Huang
# Copyright (c) 2019
###########################################################################

import os
import argparse
import torch
from encoding.utils.arg_parser_utils import CommaSeparatedSeq

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='COCO Part Segmentation')
        # model and dataset
        parser.add_argument('--norm_layer', type=str, default='bn', help='sync_bn, bn')
        parser.add_argument('--model', type=str, default='danet', help='model name (default: danet)')
        parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='coco_part', help='dataset name (default: coco_part)')
        parser.add_argument('--data-folder', type=str, default='dataset', help='training dataset folder (default: dataset')
        parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=192, help='base image size')
        parser.add_argument('--crop-size', type=int, default=256, help='crop image size')
        parser.add_argument('--train-split', type=str, default='train_market1501_cuhk03_duke_style', help='dataset train split, e.g. train, train_market1501_cuhk03_duke_style, (default: train_market1501_cuhk03_duke_style)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default=False, help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2, help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default=False, help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2, help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
        parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
        parser.add_argument('--test-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 16)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: 0.003)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default', help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None, help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False, help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--multi_scale_eval', action='store_true', default=False, help='Validation using multi scales')
        parser.add_argument('--eval_scales', type=CommaSeparatedSeq(seq_class=list, func=float), default=[1.0], help='Scales')
        parser.add_argument('--crop_eval', action='store_true', default=False, help='Multi-crop evaluation. Otherwise, the whole image is fed to network.')
        parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
        parser.add_argument('--only-val', action='store_true', default=False, help='only validation, no training')
        parser.add_argument('--only-vis', action='store_true', default=False, help='only visualization, no training')
        parser.add_argument('--only-infer', action='store_true', default=False, help='only inference, no training')
        # visualization
        parser.add_argument('--dir_of_im_to_vis', type=str, default='None', help='Directory of images, some of which will be chosen to visualize')
        parser.add_argument('--vis_save_dir', type=str, default='None', help='Directory to save visualization')
        parser.add_argument('--im_list_file_to_vis', type=str, default='None', help='Specify which images to visualize in a file')
        parser.add_argument('--max_num_vis', type=int, default=64)
        parser.add_argument('--exp_dir', type=str, default='exp', help='Directory to save experiment output (default: exp)')
        # Infer and save
        parser.add_argument('--dir_of_im_to_infer', type=str, default='None', help='All images in this directory and its sub directories will be inferred.')
        parser.add_argument('--infer_save_dir', type=str, default='None', help='Prediction will be saved to this directory, with same sub paths as original images.')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.multi_gpu = torch.cuda.device_count() > 1
        print(args)
        return args
