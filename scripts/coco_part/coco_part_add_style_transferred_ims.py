"""
cd ${DANet}
python scripts/coco_part_add_style_transferred_ims.py
"""
from copy import deepcopy
from encoding.utils.utils import read_lines, write_list

# ori_im_mask_path_file = 'datasets/coco_part/train.txt'
# save_path = 'datasets/coco_part/train_market1501_cuhk03_duke_style.txt'
# sub_dirs = ['train_market1501_style', 'train_cuhk03_style', 'train_duke_style']

# ori_im_mask_path_file = 'datasets/coco_part/train_fg.txt'
# save_path = 'datasets/coco_part/train_fg_market1501_cuhk03_duke_style.txt'
# sub_dirs = ['train_market1501_style', 'train_cuhk03_style', 'train_duke_style']

# ori_im_mask_path_file = 'datasets/coco_part/train.txt'
# save_path = 'datasets/coco_part/train_cuhk03_style.txt'
# sub_dirs = ['train_cuhk03_style']

ori_im_mask_path_file = 'datasets/coco_part/train.txt'
save_path = 'datasets/coco_part/train_duke_style.txt'
sub_dirs = ['train_duke_style']

ori_lines = read_lines(ori_im_mask_path_file)
new_lines = []
for l in ori_lines:
    parts = l.split('/')
    # The original line is also used
    new_lines.append(l)
    for sub_dir in sub_dirs:
        parts_copy = deepcopy(parts)
        parts_copy[1] = sub_dir
        new_lines.append('/'.join(parts_copy))
write_list(new_lines, save_path, '\n')
print('ori file {} lines, new file {} lines'.format(len(ori_lines), len(new_lines)))