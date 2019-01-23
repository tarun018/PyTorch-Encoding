export exp_root=${exp_root:=exp}

CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--norm_layer bn \
--train-split train_market1501_cuhk03_duke_style \
--batch-size 16 \
--test-batch-size 16 \
--exp_dir ${exp_root}/train_market1501_cuhk03_duke_style \
--dir_of_im_to_vis dataset/reid_ims_to_vis \
--max_num_vis 128
