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


base_dir = '/media/storage/glacier_dash_data/millan/step1/'
file_names = glob.glob(base_dir + 'V_*.tif')

file_names = [base_dir + 'V_{}.tif'.format(t) for t in tiles]
out_names = ['/media/storage/glacier_dash_data/indexes/{}.tif'.format(t) for t in tiles]


write_index_rasters(file_names, out_names)
print(file_names)
print(out_names)
