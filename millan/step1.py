import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
from rasterio.enums import Resampling

"""
Downsample velocity and thickness data.
"""

base_dir = '/media/storage/glacier_dash_data/millan/'
file_names = glob.glob(base_dir + 'V_*.tif')
res = 100.

tiles = {
    '1.1' : '0',
    '1.2' : '1',
    '1.3' : '2',
    '1.4' : '3',
    '1.5' : '4',
    '1.6' : '5',
    '2.1' : '6',
    '2.2' : '7',
    '2.3' : '8',
    '2.4' : '9'
}

for file_name in file_names:
    data = rioxarray.open_rasterio(file_name)
    data.data[np.isnan(data.data)] = 0.
    
    data = data.rio.reproject(
        'EPSG:3857',
        resampling=Resampling.bilinear,
        res = res
    )
    
    file_name = file_name.split('/')[-1]
    data_name = file_name.split('_')[0]
    tile_name = tiles[file_name.split('_')[1].split('-')[1]]
    out_dir = base_dir + 'step1/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    raster_name = 'V_' + tile_name + '.tif'
    #plt.imshow(data.data[0])
    #plt.show()
    data.rio.to_raster(out_dir + raster_name)

file_names = glob.glob(base_dir + 'THICKNESS_*.tif')

for file_name in file_names:
    data = rioxarray.open_rasterio(file_name)
   
    data = data.rio.reproject(
        'EPSG:3857',
        resampling=Resampling.bilinear,
        nodata=0.,
        res = res
    )
    #data.data[np.isnan(data.data)] = 0.
    #data.data[data.data == 0.] = np.nanw
    #plt.imshow(data.data[0])
    #plt.colorbar()
    #plt.show()
    
    file_name = file_name.split('/')[-1]
    data_name = file_name.split('_')[0]
    tile_name = tiles[file_name.split('_')[1].split('-')[1]]
    out_dir = base_dir + 'step1/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    raster_name = 'H_' + tile_name + '.tif'
    data.rio.to_raster(out_dir + raster_name)
