import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray
import glob
import os
import pandas as pd
import zarr
from dask.array import from_zarr

x = from_zarr("test.zarr")
print(x)
quit()

data = xr.open_zarr('test.zarr', group='0')
print(data)
quit()
base_dir = '/media/storage/glacier_dash_data/glaciers/'
dirs = glob.glob(base_dir + '/*/')

dirs = dirs[0:4]

store = zarr.DirectoryStore('test.zarr')


datasets = []
i = 0
for d in dirs:
    S = rioxarray.open_rasterio(
        d + 'dem.tif'
    )

    ds = xr.Dataset()
    ds.coords['x'] = S.x
    ds.coords['y'] = S.y
    ds['S'] = (['y', 'x'], S.data[0].astype(np.float32))

    z = ds.to_zarr(store, group=str(i), mode='w')
    i += 1






