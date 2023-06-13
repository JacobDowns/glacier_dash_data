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


index_files = ['/media/storage/glacier_dash_data/index_tiles/{}.tif'.format(i) for i in range(24)]

for i in range(24):
    print(i)

    index = rioxarray.open_rasterio(index_files[i])
    
    dhdt = rioxarray.open_rasterio(base_dir + 'step2/dhdt_{}.tif'.format(i))
    dhdt.data[dhdt.data < -1000.] = np.nan
    #print(dhdt.data)
    dhdt.data[0][index.data[0] == 0] = np.nan

    
    #plt.imshow(dhdt.data[0], vmin = -20., vmax=20.)
    #plt.colorbar()
    #plt.show()


    out_file = base_dir + 'step3/huggonet_dhdt_{}.tif'.format(i)
    dhdt.rio.to_raster(out_file, nodata = np.nan)


    #quit()
