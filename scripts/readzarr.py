import zarr

# Specify the Zarr file path and the subgroup of interest
zarr_path = "/scratch/temp/test"
subgroup_path = "features" #there is also coords but i don't actually use that for training.

# Load the Zarr array
zarr_array = zarr.open(zarr_path, mode='r')[subgroup_path]

# Print shape and dtype
print(f"Shape of the features dir: {zarr_array.shape}")
print(f"Data type of one of the feature vectors (a 224x224 patch extracted at 1.0 microns per pixel): {zarr_array[0].dtype}")