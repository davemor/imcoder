from functools import reduce
from PIL import Image, ImageStat
from pathlib import Path
import numpy as np
import pandas as pd
import shutil

import multiprocess
pool = multiprocess.Pool()

from torchvision.datasets.folder import default_loader, is_image_file


# create the output dir
excluded_dir = Path('/data/cervical_processed/excluded_256_1')
patches_dir = Path('/data/cervical_processed/patches_256_1')

image_dirs = [f for f in Path(patches_dir).iterdir() if f.is_dir()]
image_dirs = sorted(image_dirs, key=lambda p: p.name)

def f(path):
    patches_dir = Path('/data/cervical_processed/patches_256_1')
    excluded_dir = Path('/data/cervical_processed/excluded_256_1')

    def is_monochromatic(img_path):
        MONOCHROMATIC_MAX_VARIANCE = 1
        with Image.open(img_path) as im:
            pixel_var = ImageStat.Stat(im).var
        return reduce(lambda x, y: x and y < MONOCHROMATIC_MAX_VARIANCE, pixel_var, True)

    if is_monochromatic(path):
        rel_path = path.relative_to(patches_dir)
        new_path = excluded_dir / rel_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, new_path)
        return True
    else:
        return False

start_idx = 1440
img_dir_index = start_idx
for img_dir in image_dirs[start_idx:]:
    filepaths = [p for p in Path(img_dir).rglob("./*") if is_image_file(p.as_posix())]
    filepaths = sorted(filepaths, key=lambda p: p.stem)

    exclude_list = pool.map(f, filepaths)

    exclude_df = pd.DataFrame({
        "patch": filepaths,
        "excluded": exclude_list
    })

    # save the csv            
    rel_img_path = img_dir.relative_to(patches_dir)
    csv_path = excluded_dir / rel_img_path
    csv_path.mkdir(parents=True, exist_ok=True)
    full_csv_path = csv_path / 'excluded.csv'
    #print(f"Exporting to: {full_csv_path}")
    exclude_df.to_csv(full_csv_path, index=False)
    print(f"slide: {img_dir_index}, excluded: {len(exclude_df[exclude_df['excluded'] == True])} of {len(exclude_df)}\t {img_dir}")

    img_dir_index = img_dir_index + 1