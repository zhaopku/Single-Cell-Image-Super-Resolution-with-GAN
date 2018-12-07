import os
import shutil
import random
from tqdm import tqdm
from PIL import Image
src_dir = '/Users/mengzhao/DSL_celled_pic'
dst_dir = '/Users/mengzhao/ds_dapi'

def create(files, tag='train'):
	cnt = 0
	if not os.path.exists(os.path.join(dst_dir, tag)):
		os.makedirs(os.path.join(dst_dir, tag))
	for file in tqdm(files, desc=tag):
		if file.endswith('.tiff'):
			im = Image.open(os.path.join(src_dir, file))
			im.save(os.path.join(dst_dir, tag, file)+'.jpg')
			#shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, tag, file))
			cnt += 1

	print('{} {}'.format(tag, cnt))

all_files = os.listdir(src_dir)
random.shuffle(all_files)
N = len(all_files)

n_train = int(N*0.8)
n_val = int(N*0.1)
n_test = N - n_train - n_val

train_files = all_files[:n_train]
val_files = all_files[n_train:n_train+n_val]
test_files = all_files[n_train+n_val:]

create(train_files, 'train')
create(val_files, 'val')
create(test_files, 'test')
