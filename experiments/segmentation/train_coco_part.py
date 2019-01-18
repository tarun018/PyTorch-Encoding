###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

from __future__ import print_function
import os
import copy
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn import BatchNorm2d

import encoding.utils as utils
from encoding.nn import SyncBatchNorm
from encoding.nn import SegmentationMultiLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_dataset
from encoding.models import get_segmentation_model
from encoding.models import MultiEvalModule

from option_coco_part import Options
from infer_utils import vis_im_list
from file_utils import walkdir


torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        valset = get_dataset(args.dataset, split='val', mode='ms_val' if args.multi_scale_val else 'fast_val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(valset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        if args.norm_layer == 'bn':
            norm_layer = BatchNorm2d
        elif args.norm_layer == 'sync_bn':
            norm_layer = SyncBatchNorm
        else:
            raise ValueError('Invalid norm_layer {}'.format(args.norm_layer))
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone = args.backbone, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer=norm_layer,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=True,
                                       multi_dilation=[2, 4, 8],
                                       only_pam=True,
                                       )
        print(model)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
        # criterions
        self.criterion = SegmentationMultiLosses()
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.multi_gpu:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        elif args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        self.single_device_model = self.model.module if self.args.multi_gpu else self.model
        # resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            self.single_device_model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)
        self.best_pred = 0.0
        self.eval_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        self.ms_evaluator = MultiEvalModule(self.single_device_model, self.nclass, scales=self.eval_scales)
        self.metric = utils.SegmentationMetric(self.nclass)

    def save_ckpt(self, epoch, score):
        is_best = False
        if score >= self.best_pred:
            is_best = True
            self.best_pred = score
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.single_device_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, self.args, is_best)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.lr_scheduler.step()
        tbar = tqdm(self.trainloader, miniters=20)
        for i, (image, target) in enumerate(tbar):

            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image)
            if self.args.multi_gpu:
                loss = self.criterion(outputs, target)
            else:
                loss = self.criterion(list(outputs) + [target])
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            ep_log = 'ep {}'.format(epoch + 1)
            lr_log = 'lr ' + '{:.6f}'.format(self.optimizer.param_groups[0]['lr']).rstrip('0')
            loss_log = 'loss {:.3f}'.format(train_loss / (i + 1))
            tbar.set_description(', '.join([ep_log, lr_log, loss_log]))

    def validation(self, epoch):

        def _get_pred(batch_im):
            # metric.update also accepts list, so no need to gather results from multi gpus
            if self.args.multi_scale_val:
                assert len(batch_im) <= torch.cuda.device_count()
                pred = self.ms_evaluator.parallel_forward(batch_im)
            else:
                outputs = self.model(batch_im)
                pred = [out[0] for out in outputs]
            return pred

        self.model.eval()
        tbar = tqdm(self.valloader, desc='\r')
        for i, (batch_im, target) in enumerate(tbar):
            pred = _get_pred(batch_im)
            self.metric.update(target, pred)
            pixAcc, mIoU = self.metric.get()
            tbar.set_description('ep {}, pixAcc: {:.4f}, mIoU: {:.4f}'.format(epoch + 1, pixAcc, mIoU))
        return self.metric.get()

    def visualize(self, epoch):
        if not hasattr(self, 'vis_im_paths'):
            im_paths = list(walkdir(self.args.dir_of_im_to_vis, ext=self.args.ext_of_im_to_vis))
            im_paths = sorted(im_paths)
            np.random.RandomState(seed=1).shuffle(im_paths)
            self.vis_im_paths = im_paths[:self.args.num_vis]
        cfg = {
            'save_path': os.path.join(self.args.exp_dir, 'vis', 'vis_epoch{}.png'.format(epoch)),
            'multi_scale_test': self.args.multi_scale_val,
            'num_class': self.nclass,
            'scales': self.eval_scales,
            'base_size': self.args.base_size,
        }
        vis_im_list(self.single_device_model, self.vis_im_paths, cfg)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    if args.only_val:
        trainer.validation(trainer.args.start_epoch)
    else:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.visualize(epoch)
            trainer.training(epoch)
            score = trainer.best_pred
            if not trainer.args.no_val:
                pixAcc, mIoU = trainer.validation(epoch)
                score = (pixAcc + mIoU) * 0.5
            trainer.save_ckpt(epoch, score)
