import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
import zarr
from scipy.stats import binned_statistic
from multiprocessing import Pool
import geopandas as gp

# Path to zarr store
file_names = glob.glob('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/*')
glacier_indexes = [int(f.split('/')[-1]) for f in file_names]
glacier_indexes = np.array(glacier_indexes)
glacier_indexes.sort()

# Just do the first 100 glaciers
glacier_indexes = glacier_indexes[0:100]

def get_median_slope(i):
    print(i)
    try :
        store = xr.open_zarr('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/' + str(i))

        x = store['x'].data
        y = store['y'].data
        z = store['dem'].data
        mask = store['mask'].data

        # Get slope vector
        def get_slope(f):
            dx = abs(x[1] - x[0])
            dy = abs(y[1] - y[0])
            fx = np.gradient(f, dx, axis=1)
            fy = np.gradient(f, dy, axis=0)
            return fx, fy

        # Get slope in degrees
        zx, zy = get_slope(z)
        slope = np.sqrt(zx**2 + zy**2)
        # Only include area on glacier
        indexes = mask == 1.
        slope = slope[indexes].compute()
        slope = np.rad2deg(np.arctan(slope))
        d = {}
        d['index'] = i
        d['slope'] = np.median(slope)
        return d

    except Exception as e:
        print(e)

pool = Pool(processes=12)
results = pool.map(get_median_slope, glacier_indexes)

results1 = []
for r in results:
    if not r == None:
        results1.append(r)

df = pd.DataFrame(results1)
df.set_index('index')
df.sort_index()
df = df.round(3)
print(df)
