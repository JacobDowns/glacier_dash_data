import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
from project_mosaic import *


src_files = glob.glob('/media/storage/glacier_dash_data/huggonet/step1/*.tif')

base_dir = '/media/storage/glacier_dash_data/index_tiles/'
dst_files = [base_dir + '{}.tif'.format(i) for i in range(24)]
out_names = ['/media/storage/glacier_dash_data/huggonet/step2/dhdt_{}.tif'.format(i) for i in range(24)]
project_mosaic(src_files, dst_files, out_names)
