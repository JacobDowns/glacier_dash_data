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

base_dir = '/media/storage/glacier_dash_data/farinotti/'


index_files = ['/media/storage/glacier_dash_data/index_tiles/{}.tif'.format(i) for i in range(24)]

for i in range(24):
    print(i)

    index = rioxarray.open_rasterio(index_files[i])
    
    H = rioxarray.open_rasterio(base_dir + 'step2/H_{}.tif'.format(i))

    print(H.data.max())
    H.data[0][index.data[0] == 0] = np.nan
    out_file = base_dir + 'step3/farinotti_H_{}.tif'.format(i)
    H.rio.to_raster(out_file, nodata = np.nan)


    plt.imshow(H.data[0])
    plt.colorbar()
    plt.show()
    #quit()
