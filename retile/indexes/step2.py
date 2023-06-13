import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
from compute_indexes import *

"""
Downsample velocity and thickness data.
"""

tiles = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6','2.1', '2.2', '2.3', '2.4']

file_names = ['/media/storage/glacier_dash_data/indexes/{}.tif'.format(t) for t in tiles]


data = rioxarray.open_rasterio(file_names[0])
data.data[data.data > 0] = 1
print(data)

plt.imshow(data.data[0])
plt.show()
