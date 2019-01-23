export exp_root=${exp_root:=exp}

CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--dir_of_im_to_vis dataset/reid_ims_to_vis \
--vis_save_dir ${exp_root}/vis_reid_ims \
--resume ${exp_root}/EANet_paper_ps_model/model_best.pth.tar \
--only-vis \
--max_num_vis 128