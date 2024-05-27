# This script rglobs images from a directory tree and reports if they are monochromatic

from functools import reduce
from PIL import Image, ImageStat

MONOCHROMATIC_MAX_VARIANCE = 1

def is_monochromatic(img):
    pixel_var = ImageStat.Stat(img).var
    return reduce(lambda x, y: x and y < MONOCHROMATIC_MAX_VARIANCE, pixel_var, True), pixel_var

black_img = '/data/cervical_processed/patches_256_1/0/normal/00000000-IC-CX-00001-01-0-0-1-256.png'
white_img = '/data/cervical_processed/patches_256_1/0/normal/00002910-IC-CX-00001-01-3072-17664-1-256.png'
white2_img = '/data/cervical_processed/patches_256_1/0/normal/00000103-IC-CX-00001-01-4864-512-1-256.png'
tissue_img = '/data/cervical_processed/patches_256_1/0/normal/00003303-IC-CX-00001-01-6912-19968-1-256.png'
tissue_img2 = '/data/cervical_processed/patches_256_1/0/normal/00004045-IC-CX-00001-01-3328-24576-1-256.png'

image_paths = [black_img, white_img, white2_img, tissue_img, tissue_img2]
for p in image_paths:
    img = Image.open(p)
    print(is_monochromatic(img))