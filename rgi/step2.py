import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
import geopandas as gp

base_dir = '/media/storage/glacier_dash_data/rgi/'
data = gp.read_file(base_dir + 'step1/all_rgi.shp')
#indexes = data['Area'] > 10
#data = data[indexes]
data['index'] = np.arange(len(data)) + 1
data.to_file(base_dir + 'step2/rgi.shp')


