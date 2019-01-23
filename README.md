# About

This is a fork from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) (refer to it for the original README). In this repository, we use [DANet](https://github.com/junfu1115/DANet) to train a part segmentation model on [COCO Densepose Dataset](http://densepose.org/), for use in our person re-identification paper [EANet](https://github.com/huanghoujing/EANet).

# Installation

First install prerequsites [as this instruction](Install_Prerequisite_for_Pytorch_Encoding.md).

Then clone this project
```bash
git clone https://github.com/huanghoujing/PyTorch-Encoding.git
```

# Dataset

We transform COCO Densepose Dataset as described in paper [EANet](https://github.com/huanghoujing/EANet). For details, please refer to `scripts/coco_part`. **Note** that import and data paths in these scripts are not re-arranged and will cause error when executed. 

The resulting segmentation dataset can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1Mm2gWO-Xg3wiyCd6SEAWaA#list/path=%2Fsharelink1629242940-281396814376268%2FEANet%2Fdataset%2Fcoco&parentPath=%2Fsharelink1629242940-281396814376268) or [Google Drive](https://drive.google.com/drive/folders/1gITlG2MfhJXUpfEPt6ohJCBgqigchabW).

Prepare the dataset to have following structure
```
${project_dir}/dataset/coco_part
    images
    masks_7_parts
    masks -> masks_7_parts  # This is a link created by `ln -s masks_7_parts masks`
    train.txt
    train_market1501_cuhk03_duke_style.txt
    val.txt
```

# Examples

Download our trained model ([Baidu Cloud](https://pan.baidu.com/s/1Mm2gWO-Xg3wiyCd6SEAWaA#list/path=%2Fsharelink1629242940-281396814376268%2FEANet%2Fpart_segmentation_model&parentPath=%2Fsharelink1629242940-281396814376268) or [Google Drive](https://drive.google.com/drive/folders/1suBZk1WhpiS5PdB3GFySEEkC2FjzPQpL)) to `${project_dir}/exp/EANet_paper_ps_model/model_best.pth.tar`.

## Infer and Visualize Your Images

Specify your image directory `dir_of_im_to_vis` and an output directory `vis_save_dir` in the following command. It will select at most `max_num_vis` images to infer and save the visualization.

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--dir_of_im_to_vis YOUR_IMAGE_DIRECTORY \
--vis_save_dir OUTPUT_DIRECTORY \
--resume exp/EANet_paper_ps_model/model_best.pth.tar \
--only-vis \
--max_num_vis 128
```

`misc/example_visualization.png` is an example result.

## Infer and Save Prediction

Specify your image directory `dir_of_im_to_infer` and an output directory `infer_save_dir` in the following command. Prediction will be saved to `infer_save_dir`, with same sub paths as original images.

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--resume exp/EANet_paper_ps_model/model_best.pth.tar \
--only-infer \
--dir_of_im_to_infer YOUR_IMAGE_DIRECTORY \
--infer_save_dir OUTPUT_DIRECTORY
```

For each image, the prediction is saved as a single-channel PNG image, with the same resolution as the input image. Each pixel value of the output image is its part label. Refer to [this link](https://github.com/huanghoujing/EANet#part-segmentation-label-format) for part index. Optionally, you can use script `experiments/coco_part/colorize_pred_mask.py` to colorize the predicted masks for visualization.

## Validate on COCO Part Val Set

The following command validates on val set, with single scale and flipping, but without cropping. You should get result `pixAcc: 0.9034, mIoU: 0.6670`.

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--resume exp/EANet_paper_ps_model/model_best.pth.tar \
--only-val
```


## Training

Since person images are much smaller compared with other segmentation tasks, we can use a single GPU while maintaining a large batch size. For example, when we set batch size to 16, the GPU usage is about 5600MB.

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--norm_layer bn \
--train-split train \
--batch-size 16 \
--test-batch-size 16 \
--exp_dir exp/train
```

You can also try multi GPUs and synchronized BN by setting `norm_layer` to `sync_bn`

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python experiments/coco_part/train.py \
--norm_layer sync_bn \
--train-split train \
--batch-size 16 \
--test-batch-size 16 \
--exp_dir exp/train
```

# Citation

If you find our work useful, please kindly cite our paper:
```
@article{huang2018eanet,
  title={EANet: Enhancing Alignment for Cross-Domain Person Re-identification},
  author={Huang, Houjing and Yang, Wenjie and Chen, Xiaotang and Zhao, Xin and Huang, Kaiqi and Lin, Jinbin and Huang, Guan and Du, Dalong},
  journal={arXiv preprint arXiv:1812.11369},
  year={2018}
}
```