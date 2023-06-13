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
data0 = gp.read_file(base_dir + '01_rgi60_Alaska.shp')
data1 = gp.read_file(base_dir + '02_rgi60_WesternCanadaUS.shp')

data = pd.concat([data0, data1])
data['index'] = np.arange(len(data))
data.to_file(base_dir + 'step1/all_rgi.shp', index=False)
print(data)


