from PIL import Image
import numpy as np

im = Image.open('./sample.jpg')
im2 = Image.open('./sample.tiff')

im_arr = np.asarray(im)
im_arr2 = np.asarray(im2)

diff = np.asarray(im) - np.asarray(im2)

print(np.sum(diff))

im.save('sample2.jpg')