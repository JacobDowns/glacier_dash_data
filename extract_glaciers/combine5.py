import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
import zarr
from datatree import DataTree, open_datatree


def combine(tile_index):

    store = zarr.DirectoryStore(
        '/media/storage/glacier_dash_data/netcdf_tiles/{}.zarr'.format(tile_index)
    )
    
    base_dir = '/media/jake/Backup/raster_data/'
    
    index_raster = rioxarray.open_rasterio(
        base_dir + 'index/{}.tif'.format(tile_index)
    )

    gindexes = np.unique(index_raster.data[0])[1:1000]
    base_dir = '/media/storage/glacier_dash_data/glaciers/'
    datasets = []
    
    for index in gindexes:
        print(index)
        d = base_dir + str(index) + '/'
        #rasters = glob.glob(d + '/*.tif')

        ds = xr.Dataset()
        
        S = rioxarray.open_rasterio(
            d + 'mask.tif'
        )

        ds = xr.Dataset()
        ds.coords['x'] = S.x
        ds.coords['y'] = S.y
        ds['S'] = (['y', 'x'], S.data[0].astype(np.float32))

        datasets.append(ds)


    data = xr.merge(datasets)

    plt.imshow(data.S)
    plt.colorbar()
    plt.show()
    print(data)
     
combine(0)

            
