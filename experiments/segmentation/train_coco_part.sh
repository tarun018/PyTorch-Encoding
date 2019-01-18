export anaconda_home=/mnt/data-1/data/houjing.huang/Software/anaconda2_encoding
export PATH=${anaconda_home}/bin:${PATH}
export LD_LIBRARY_PATH=${anaconda_home}/lib:${LD_LIBRARY_PATH}

#export CUDA_HOME=/mnt/data-1/data/houjing.huang/Software/cuda-9.0
#export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export PATH=${PWD}/bin:${PATH}

#python setup.py install

CUDA_VISIBLE_DEVICES=2,3 \
python experiments/segmentation/train_coco_part.py \
--dir_of_im_to_vis /mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet_Data/dataset/market1501/Market-1501-v15.09.15/bounding_box_train \
--batch-size 4 \
--norm_layer bn