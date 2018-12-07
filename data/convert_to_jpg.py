import os
import shutil
import random
from tqdm import tqdm
from PIL import Image
src_dir = '/Users/mengzhao/cell-sr/40x_pic/EGFP'
dst_dir = '/Users/mengzhao/cell-sr/40x_pic/jpg/EGFP'

def create(files):
	cnt = 0
	for file in tqdm(files):
		if file.endswith('.tiff'):
			im = Image.open(os.path.join(src_dir, file))
			if im.size != (72, 72):
				continue
			im.save(os.path.join(dst_dir, file)+'.jpg')
			#shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, tag, file))
			cnt += 1

	print('{}'.format(cnt))
if not os.path.exists(dst_dir):
	os.makedirs(dst_dir)
all_files = os.listdir(src_dir)
random.shuffle(all_files)
N = len(all_files)

create(all_files)
