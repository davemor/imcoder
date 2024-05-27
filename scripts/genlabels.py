# this script goes over all the images in each of the folders
# and works out what their full patch label is.
# Patch labels are the <slide_subcategory>_<patch_label>
# I.e. each slide has two labels

# the other way to do it is to get all the patch images
# then load the slides index
# remove any patch from the index that doesn't exist
# then save the index somewhere else
# That way all of the information is retained.
# Let us do that.

from pathlib import Path
from torchvision.datasets.folder import is_image_file
from patch_set import PatchSet
from tqdm import tqdm

patches_dir = Path('/data/cervical_processed/patches_256_1')

image_dirs = [f for f in Path(patches_dir).iterdir() if f.is_dir()]
image_dirs = sorted(image_dirs, key=lambda p: p.name)

# load in the patchsets, sort them by integer order
patchsets_dir = Path('/data/cervical_processed/index_256_1')
patchset_paths = [f for f in patchsets_dir.iterdir() if f.is_dir()]
patchset_paths = sorted(patchset_paths, key=lambda p: int(p.name))
patchsets = []
print("Loading patch sets...")
for path in tqdm(patchset_paths):
    pset = PatchSet.load(path)
    patchsets.append(pset)

    # also add the excluded column to the df
    pset.df['excluded'] = True

print("Filtering patchsets...")
start_idx = 0  # this is so it can be restarted simply
img_dir_index = start_idx
# count = 1
for img_dir in tqdm(image_dirs[start_idx:]):
    filepaths = [p for p in Path(img_dir).rglob("./*") if is_image_file(p.as_posix())]
    filepaths = sorted(filepaths, key=lambda p: p.stem)
    for img_path in filepaths:
        index_of_patch_in_patchset = int(img_path.stem[:8])
        index_of_patchset = int(img_path.parent.parent.stem)
        # print(index_of_patchset, index_of_patch_in_patchset)
        patchsets[index_of_patchset].df.at[index_of_patch_in_patchset, 'excluded'] = False

print("Saving filtered patchsets...")
filtered_index_dir = Path('/data/cervical_processed/filtered_index_256_1')
for idx, pset in tqdm(enumerate(patchsets)):
    pset.df = pset.df[pset.df.excluded == False]
    pset.df = pset.df.drop('excluded', axis=1)
    pset.save(filtered_index_dir / str(idx))