import os


hdfs_home = "hdfs://hobot-bigdata-aliyun/user/houjing.huang"


def put(src, dest):
    dest = os.path.join(hdfs_home, dest)
    os.system('hadoop fs -mkdir -p {}'.format(os.path.dirname(dest)))
    os.system('hadoop fs -put {} {}'.format(src, dest))
    print('Put {} to {}'.format(src, dest))
    os.system('hadoop fs -ls -h {}'.format(dest))

src_dest = [
    ('/mnt/data-1/data/houjing.huang/Software/anaconda2_encoding.tar.gz', 'Software/'),
    # ('/mnt/data-1/data/houjing.huang/Software/cuda-9.0.tar.gz', 'Software/'),
    # ('/mnt/data-1/data/houjing.huang/Project/PyTorch-Encoding/dataset/reid_ims_to_vis.tar.gz', 'Dataset/encoding/'),
    # ('/mnt/data-1/data/houjing.huang/Project/PyTorch-Encoding-Data/exp/EANet_paper_ps_model/model_best.pth.tar', 'Exp/encoding/EANet_paper_ps_model/model_best.pth.tar'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/partial_reid/Partial-REID_Dataset.tar.gz', 'Dataset/partial_reid/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/partial_ilids/Partial_iLIDS.zip', 'Dataset/partial_ilids/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/images.tar.gz', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/masks_7_parts.tar.gz', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/masks_14_parts.tar.gz', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/masks_fg.tar.gz', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/im_name_to_kpt.pkl', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/im_name_to_h_w.pkl', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/train.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/train_fg.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/val.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/val_fg.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/train_market1501_style.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/train_cuhk03_style.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/train_duke_style.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/train_market1501_cuhk03_duke_style.txt', 'Dataset/eanet/coco/'),
    # ('/mnt/data-1/data/houjing.huang/Project/EANet_cluster/EANet/dataset/coco/train_fg_market1501_cuhk03_duke_style.txt', 'Dataset/eanet/coco/'),
]

for src, dest in src_dest:
    put(src, dest)
