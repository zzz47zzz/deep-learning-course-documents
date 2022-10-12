import os
import numpy as np
import os
import PIL
from PIL import Image
from multiprocessing import Pool

def pad_image(img_path, output_path, pad_size=[8, 8, 8, 8], buckets=None):
    """Pads image with pad size and with buckets

    Args:
        img_path: (string) path to image
        output_path: (string) path to output image
        pad_size: list of 4 ints
        buckets: ascending ordered list of sizes, [(width, height), ...]

    """
    top, left, bottom, right = pad_size
    old_im = Image.open(img_path)
    old_size = (old_im.size[0] + left + right, old_im.size[1] + top + bottom)
    new_size = get_new_size(old_size, buckets)
    new_im = Image.new("RGB", new_size, (255, 255, 255))
    new_im.paste(old_im, (left, top))
    new_im.save(output_path)

def get_new_size(old_size, buckets):
    """Computes new size from buckets

    Args:
        old_size: (width, height)
        buckets: list of sizes

    Returns:
        new_size: original size or first bucket in iter order that matches the
            size.

    """
    if buckets is None:
        return old_size
    else:
        w, h = old_size
        for (w_b, h_b) in buckets:
            if w_b >= w and h_b >= h:
                return w_b, h_b

        return old_size

input_image_dir = '../data/210618/formulas_images_no_padding/'
output_image_dir = '../data/210618/formulas_images/'
if not os.path.isdir(output_image_dir):
    os.mkdir(output_image_dir)

img_path_list = os.listdir(input_image_dir)

buckets = [
        [60, 40], [80, 40], [80, 60], [140, 60], [140, 80], 
        [240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
        [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
        [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
        [1000, 400], [1200, 200], [1600, 200], [1600, 1600]
        ]

for img_path in img_path_list:
    pad_image(input_image_dir + img_path, output_image_dir + os.path.basename(img_path), buckets=buckets)