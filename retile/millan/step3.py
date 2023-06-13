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
Check to make sure tiles look correct. 
"""



base_dir = '/media/storage/glacier_dash_data/'

#H = rioxarray.open_rasterio(base_dir + 'millan/step3/H_0.tif')
#plt.imshow(H.data[0])
#plt.show()
#quit()

index_files = ['/media/storage/glacier_dash_data/index_tiles/{}.tif'.format(i) for i in range(24)]
H_files = ['/media/storage/glacier_dash_data/millan/step2/H_{}.tif'.format(i) for i in range(24)]
V_files = ['/media/storage/glacier_dash_data/millan/step2/V_{}.tif'.format(i) for i in range(24)]

#file_names = glob.glob('/media/storage/glacier_dash_data/millan/step2/H_*.tif'

for i in range(24):
    print(i)

    index = rioxarray.open_rasterio(index_files[i])

    try:
        H = rioxarray.open_rasterio(H_files[i])
        if H.data.max() > 0.:
            H.data[0][index.data[0] == 0] = np.nan
            out_file = base_dir + 'millan/step3/millan_H_{}.tif'.format(i)
            H.rio.to_raster(out_file, nodata = np.nan)
    except:
        print('H error')

    try:
        V = rioxarray.open_rasterio(V_files[i])
        if V.data.max() > 0.:
            V.data[0][index.data[0] == 0] = np.nan
            out_file = base_dir + 'millan/step3/millan_V_{}.tif'.format(i)
            V.rio.to_raster(out_file, nodata = np.nan)
    except:
        print('v error')
