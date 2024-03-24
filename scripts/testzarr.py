import zarr

import numpy as np

arr = np.random.random((10, 10))

store = zarr.DirectoryStore('/scratch/temp/test')
root = zarr.open_group(store=store, mode='w')

key = 'features'

# chunk shape should be base on the access patterns so (batch_size, features)

dset = root.create_dataset(key, shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
dset[:] = arr