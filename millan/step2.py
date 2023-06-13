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
Downsample velocity and thickness data.
"""


src_files = glob.glob('/media/storage/glacier_dash_data/millan/step1/H_*.tif')

base_dir = '/media/storage/glacier_dash_data/index_tiles/'
dst_files = [base_dir + '{}.tif'.format(i) for i in range(24)]
out_names = ['/media/storage/glacier_dash_data/millan/step2/H_{}.tif'.format(i) for i in range(24)]
project_mosaic(src_files, dst_files, out_names)

src_files = glob.glob('/media/storage/glacier_dash_data/millan/step1/V_*.tif')
out_names = ['/media/storage/glacier_dash_data/millan/step2/V_{}.tif'.format(i) for i in range(24)]
project_mosaic(src_files, dst_files, out_names)
