from PIL import Image
import numpy as np

im = Image.open('./sample.tiff')

im.save('sample.jpg', quality=100)

