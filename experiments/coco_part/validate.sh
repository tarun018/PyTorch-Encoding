export exp_root=${exp_root:=exp}

CUDA_VISIBLE_DEVICES=0 \
python experiments/coco_part/train.py \
--resume ${exp_root}/EANet_paper_ps_model/model_best.pth.tar \
--only-val