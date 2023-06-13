import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
from project_mosaic import *

"""
Reproject Farinotti data.
"""

base_dir = '/media/storage/glacier_dash_data/huggonet/'
file_names = glob.glob(base_dir + '*.tif')

for file_name in file_names:
    print(file_name)

    name = file_name.split('/')[-1]
    
    data = rioxarray.open_rasterio(file_name)
    #data.data[np.isnan(data.data)] = 0.
    #plt.imshow(data.data[0])
    #plt.colorbar()
    #plt.show()
    
    data = data.rio.reproject(
        'EPSG:3857',
        resampling=Resampling.bilinear,
        nodata = -9.999e3
    )

    data.rio.to_raster(base_dir + 'step1/' + name)
    #plt.imshow(data.data[0])
    #plt.colorbar()
    #plt.show()
