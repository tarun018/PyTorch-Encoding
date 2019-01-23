export exp_root=${exp_root:=exp}

CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--resume ${exp_root}/EANet_paper_ps_model/model_best.pth.tar \
--only-infer \
--dir_of_im_to_infer dataset/reid_ims_to_vis \
--infer_save_dir ${exp_root}/infer_reid_ims_to_vis