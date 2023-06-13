import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
from raster_stats import *
import traceback

base_dir = '/media/jake/Backup/raster_data/'
file_names = glob.glob(base_dir + 'H_farinotti/*.tif')
stats_data = []


for file_name in file_names:
    try:
        name = file_name.split('/')[-1]
        print(name)

        index_raster = rioxarray.open_rasterio(base_dir + 'index/{}'.format(name))
        data_raster = rioxarray.open_rasterio(base_dir + 'H_farinotti/{}'.format(name))
        
        data = data_raster.data[0]
        data[data < 0.01] = np.nan
        
        df = get_stats(index_raster.data[0], data)
        df = df[1:]
        stats_data.append(df)
    except:
        traceback.print_exc()


df = pd.concat(stats_data)
df = df.sort_values('indexes')
print(df)

df.to_csv('H_farinotti_stats.csv', index=False)

