###########################################################################
# Created by: Houjing Huang
# Copyright (c) 2019
###########################################################################
"""
- Allow single GPU.
  - For single GPU, vanilla BN is used. For multi GPUs, Sync-BN is used.
  - For single GPU, we have to manually move the data to GPU. For multi GPUs, DataParallel will scatter data to multi GPUs.
- No CPU mode.
- Visualize an image list during training.
- Infer and save prediction for image list.
- In single-scale validation, use full image instead of center crop.
- Allow multi-scale eval.
- Allow no cropping in multi-scale eval.
- The model is DANet.
"""

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
from encoding.datasets import get_dataset, test_batchify_fn
from encoding.models import get_segmentation_model
from encoding.models.danet_base import MultiEvalModule

from option import Options
from encoding.utils.infer_utils import vis_im_list
from encoding.utils.infer_utils import infer_and_save_im_list
from encoding.utils.file_utils import walkdir
from encoding.utils.file_utils import read_lines


print('[PYTORCH VERSION]:', torch.__version__)
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
        valset = get_dataset(args.dataset, split='val', mode='ms_val' if args.multi_scale_eval else 'fast_val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        if self.args.multi_scale_eval:
            kwargs['collate_fn'] = test_batchify_fn
        self.valloader = data.DataLoader(valset, batch_size=args.test_batch_size, drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        if args.norm_layer == 'bn':
            norm_layer = BatchNorm2d
        elif args.norm_layer == 'sync_bn':
            assert args.multi_gpu, "SyncBatchNorm can only be used when multi GPUs are available!"
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
        else:
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
            if not args.ft and not (args.only_val or args.only_vis or args.only_infer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {}), best_pred {}"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_pred']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)
        self.best_pred = 0.0

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

            if not self.args.multi_gpu:
                image = image.cuda()
                target = target.cuda()
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            outputs = self.model(image)
            if self.args.multi_gpu:
                loss = self.criterion(outputs, target)
            else:
                loss = self.criterion(*(outputs + (target,)))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            ep_log = 'ep {}'.format(epoch + 1)
            lr_log = 'lr ' + '{:.6f}'.format(self.optimizer.param_groups[0]['lr']).rstrip('0')
            loss_log = 'loss {:.3f}'.format(train_loss / (i + 1))
            tbar.set_description(', '.join([ep_log, lr_log, loss_log]))

    def validation(self, epoch):

        def _get_pred(batch_im):
            with torch.no_grad():
                # metric.update also accepts list, so no need to gather results from multi gpus
                if self.args.multi_scale_eval:
                    assert len(batch_im) <= torch.cuda.device_count(), "Multi-scale testing only allows batch size <= number of GPUs"
                    scattered_pred = self.ms_evaluator.parallel_forward(batch_im)
                else:
                    outputs = self.model(batch_im)
                    scattered_pred = [out[0] for out in outputs] if self.args.multi_gpu else [outputs[0]]
            return scattered_pred

        # Lazy creation
        if not hasattr(self, 'ms_evaluator'):
            self.ms_evaluator = MultiEvalModule(self.single_device_model, self.nclass, scales=self.args.eval_scales, crop=self.args.crop_eval)
            self.metric = utils.SegmentationMetric(self.nclass)
        self.model.eval()
        tbar = tqdm(self.valloader, desc='\r')
        for i, (batch_im, target) in enumerate(tbar):
            # No need to put target to GPU, since the metrics are calculated by numpy.
            # And no need to put data to GPU manually if we use data parallel.
            if not self.args.multi_gpu and not isinstance(batch_im, (list, tuple)):
                batch_im = batch_im.cuda()
            scattered_pred = _get_pred(batch_im)
            scattered_target = []
            ind = 0
            for p in scattered_pred:
                target_tmp = target[ind:ind + len(p)]
                # Multi-scale testing. In fact, len(target_tmp) == 1
                if isinstance(target_tmp, (list, tuple)):
                    assert len(target_tmp) == 1
                    target_tmp = torch.stack(target_tmp)
                scattered_target.append(target_tmp)
                ind += len(p)
            self.metric.update(scattered_target, scattered_pred)
            pixAcc, mIoU = self.metric.get()
            tbar.set_description('ep {}, pixAcc: {:.4f}, mIoU: {:.4f}'.format(epoch + 1, pixAcc, mIoU))
        return self.metric.get()

    def visualize(self, epoch):
        if (self.args.dir_of_im_to_vis == 'None') and (self.args.im_list_file_to_vis == 'None'):
            return
        if not hasattr(self, 'vis_im_paths'):
            if self.args.dir_of_im_to_vis != 'None':
                print('=> Visualize Dir {}'.format(self.args.dir_of_im_to_vis))
                im_paths = list(walkdir(self.args.dir_of_im_to_vis, exts=['.jpg', '.png']))
            else:
                print('=> Visualize Image List {}'.format(self.args.im_list_file_to_vis))
                im_paths = read_lines(self.args.im_list_file_to_vis)
            print('=> Save Dir {}'.format(self.args.vis_save_dir))
            im_paths = sorted(im_paths)
            # np.random.RandomState(seed=1).shuffle(im_paths)
            self.vis_im_paths = im_paths[:self.args.max_num_vis]
        cfg = {
            'save_path': os.path.join(self.args.vis_save_dir, 'vis_epoch{}.png'.format(epoch)),
            'multi_scale': self.args.multi_scale_eval,
            'crop': self.args.crop_eval,
            'num_class': self.nclass,
            'scales': self.args.eval_scales,
            'base_size': self.args.base_size,
        }
        vis_im_list(self.single_device_model, self.vis_im_paths, cfg)

    def infer_and_save(self, infer_dir, infer_save_dir):
        print('=> Infer Dir {}'.format(infer_dir))
        print('=> Save Dir {}'.format(infer_save_dir))
        sub_im_paths = list(walkdir(infer_dir, exts=['.jpg', '.png'], sub_path=True))
        im_paths = [os.path.join(infer_dir, p) for p in sub_im_paths]
        # NOTE: Don't save result as JPEG, since it causes aliasing.
        save_paths = [os.path.join(infer_save_dir, p.replace('.jpg', '.png')) for p in sub_im_paths]
        cfg = {
            'multi_scale': self.args.multi_scale_eval,
            'crop': self.args.crop_eval,
            'num_class': self.nclass,
            'scales': self.args.eval_scales,
            'base_size': self.args.base_size,
        }
        infer_and_save_im_list(self.single_device_model, im_paths, save_paths, cfg)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    if args.only_val:
        trainer.validation(trainer.args.start_epoch)
    elif args.only_vis:
        trainer.visualize(trainer.args.start_epoch)
    elif args.only_infer:
        trainer.infer_and_save(args.dir_of_im_to_infer, args.infer_save_dir)
    else:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            score = trainer.best_pred
            if not trainer.args.no_val:
                pixAcc, mIoU = trainer.validation(epoch)
                score = (pixAcc + mIoU) * 0.5
            trainer.save_ckpt(epoch, score)
            trainer.visualize(epoch)
