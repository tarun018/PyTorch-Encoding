import numpy as np
import cv2
import torch
from vis_utils import read_im, make_im_grid, save_im, mask_to_im
from encoding.models.base import MultiEvalModule, module_inference


def infer_one_im(model, im_path, cfg):
    """
    Returns:
        pred: numpy array with shape [im_w, im_h], uint8
    """
    model.eval()
    im = read_im(im_path, convert_rgb=True, resize_h_w=None, transpose=False)
    H, W = im.shape[0], im.shape[1]
    im = im / 255.
    im = im - np.array([.485, .456, .406])[np.newaxis, np.newaxis, :]
    im = im / np.array([.229, .224, .225])[np.newaxis, np.newaxis, :]
    if cfg['multi_scale_test']:
        evaluator = MultiEvalModule(model, cfg['num_class'], scales=cfg['scales'])
        with torch.no_grad():
            im = torch.from_numpy(im).float().cuda().unsqueeze(0)
            pred = evaluator.forward(im)
    else:
        scale = cfg['base_size'] / min(im.shape[0], im.shape[1])
        resize_h_w = (int(im.shape[0] * scale), int(im.shape[1] * scale))
        im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().cuda().unsqueeze(0)
        with torch.no_grad():
            pred = module_inference(model, im, flip=True)
    pred = pred.detach().cpu().numpy()[0]
    pred = cv2.resize(np.argmax(pred, 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return pred


def vis_one_im(im_path, pred, resize_w_h=(64, 128)):
    """
    Args:
        pred: np array, shape [H, W], integer values
    Returns:
        im, pred: np array, shape [3, h, w], uint8
    """
    im = read_im(im_path, convert_rgb=True, resize_h_w=resize_w_h[::-1], transpose=True)
    pred = mask_to_im(pred, transpose=False)
    pred = cv2.resize(pred, tuple(resize_w_h), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    return im, pred


def vis_im_list(model, im_paths, cfg):
    # Infer a list of images and visualize
    vis_ims_preds = []
    for im_p in im_paths:
        pred = infer_one_im(model, im_p, cfg)
        vis_ims_preds += vis_one_im(im_p, pred)
    n_cols = 16
    n_rows = int(np.ceil(len(vis_ims_preds) / n_cols))
    vis_im = make_im_grid(vis_ims_preds, n_rows, n_cols, 4, 255)
    save_im(vis_im, cfg['save_path'], transpose=True, check_bound=True)
