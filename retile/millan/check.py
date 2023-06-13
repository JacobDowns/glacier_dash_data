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

V_files = ['/media/storage/glacier_dash_data/millan/step1/V_{}.tif'.format(i) for i in range(24)]
#V_files = ['/media/storage/glacier_dash_data/millan/step1/V_{}.tif'.format(i) for i in range(10)]

for i in range(24):
    V = rioxarray.open_rasterio(V_files[i])
    #V = rioxarray.open_rasterio(V_files[i])


    plt.imshow(V.data[0])
    plt.colorbar()
    plt.show()
    #quit()

    """
    print(V.rio.transform())
    print(H.rio.crs)
    print(H)
    print(H.rio.bounds())
    quit()
    print(H)
    print(V)
    print()
    plt.imshow(H.data[0])
    plt.colorbar()
    plt.show()
    """

