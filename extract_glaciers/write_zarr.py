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
from scipy.stats import binned_statistic

#2600

for i in range(1,1000):
    print(i)
    store = xr.open_zarr('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/' + str(i))
    print(store)

    mask = store['mask'].data
    dhdt = store['dh'].data
    z = store['dem'].data
    indexes = (mask > 0.)
    z = z[indexes]
    dhdt = dhdt[indexes]

    #zindexes = np.argsort(z)
    #z = z[zindexes]
    #dhdt = dhdt[zindexes]

    result = binned_statistic(z, dhdt, statistic='mean', bins=20)

    plt.plot(result.statistic)
    plt.show()

quit()
data = open_datatree('/media/storage/glacier_dash_data/netcdf_tiles/11.zarr', engine='zarr')
print(dir(data))
quit()

def combine(tile_index):

    store = zarr.DirectoryStore(
        '/media/storage/glacier_dash_data/netcdf_tiles/{}.zarr'.format(tile_index)
    )
    
    base_dir = '/media/jake/Backup/raster_data/'
    
    index_raster = rioxarray.open_rasterio(
        base_dir + 'index/{}.tif'.format(tile_index)
    )

    gindexes = np.unique(index_raster.data[0])[1:]
    base_dir = '/media/storage/glacier_dash_data/glaciers/'
    datasets = {}
    
    for index in gindexes:
        print(index)
        d = base_dir + str(index) + '/'
        rasters = glob.glob(d + '/*.tif')

        ds = xr.Dataset()
        
        j = 0
        for f in rasters:
            name = f.split('/')[-1][:-4]
            data = rioxarray.open_rasterio(f)

            if j == 0:
                ds.coords['x'] = data.x
                ds.coords['y'] = data.y

            ds[name] = (['y', 'x'], data.data[0].astype(np.float32))

        datasets[str(index)] = ds


    dt = DataTree.from_dict(datasets)
    dt.to_zarr(store, mode='w'
     
combine(12)
quit()

            
