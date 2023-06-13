import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray
import glob
import os
import pandas as pd
import zarr
from dask.array import from_zarr
from netCDF4 import Dataset
from datatree import DataTree, open_datatree

data = open_datatree('test.nc')
print(data)
quit()

base_dir = '/media/storage/glacier_dash_data/glaciers/'
dirs = glob.glob(base_dir + '/*/')
dirs = dirs[0:4]
i = 0

datasets = {}
with Dataset('multiple_datasets.nc', 'w') as file:
    for d in dirs:
        S = rioxarray.open_rasterio(
            d + 'dem.tif'
        )

        ds = xr.Dataset()
        ds.coords['x'] = S.x
        ds.coords['y'] = S.y
        ds['S'] = (['y', 'x'], S.data[0].astype(np.float32))

        datasets[str(i)] = ds
        i += 1

dt = DataTree.from_dict(datasets)
dt.to_netcdf("test.nc", format="NETCDF4")




