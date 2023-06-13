import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
from raster_stats import *
import traceback

df = pd.read_csv('alt_metric_stats.csv')
df1 =  pd.read_csv('thinning_metric_stats.csv')

m0 = df['metric']
m1 = df1['metric']

plt.plot(m0, m1, 'ro')
plt.xlim([-2,1.5])
plt.show()


plt.hist(df['metric'], bins= np.linspace(-5,5,200))
plt.hist(df1['metric'], bins= np.linspace(-5,5,200))

plt.show()
print(df['metric'].mean())

quit()

base_dir = '/media/jake/Backup/raster_data/'
file_names = glob.glob(base_dir + 'dhdt/*.tif')
stats_data = []
"""
for file_name in file_names:
    try:
        name = file_name.split('/')[-1]
        print(name)

        index_raster = rioxarray.open_rasterio(base_dir + 'index/{}'.format(name))
        dhdt_raster = rioxarray.open_rasterio(base_dir + 'dhdt/{}'.format(name))
        dem_raster = rioxarray.open_rasterio(base_dir + 'dem/{}'.format(name))

        plt.imshow(dem_raster[0])
        plt.colorbar()
        plt.show()
        quit()

        
        data = data_raster.data[0]
        #data[data < 0.01] = np.nan
        
        df = get_stats(index_raster.data[0], data)
        df = df[1:]
        stats_data.append(df)
    except:
        traceback.print_exc()
"""

for file_name in file_names:
    try:
        name = file_name.split('/')[-1]
        print(name)

        index_raster = rioxarray.open_rasterio(base_dir + 'index/{}'.format(name))
        dhdt_raster = rioxarray.open_rasterio(base_dir + 'dhdt/{}'.format(name))
        dem_raster = rioxarray.open_rasterio(base_dir + 'dem/{}'.format(name))


        dhdt = dhdt_raster.data[0]
        m = dhdt * dem_raster.data[0]

    
        df0 = get_stats(index_raster.data[0], dhdt)
        df0 = df0[1:]

        df1 = get_stats(index_raster.data[0], m)
        df1 = df1[1:]

        df2 = get_stats(index_raster.data[0], dem_raster.data[0])
        df2 = df2[1:]

        indexes = df2['indexes'].to_numpy().astype(int)


        z_min = df2['min']
        z_max = df2['max']
        z_r = z_max - z_min

        metric = df1['sum'] / (z_r * df0['sum']) - (z_min / z_r)
        df1['metric'] = metric
        stats_data.append(df1)
    except:
        traceback.print_exc()



df = pd.concat(stats_data)
df = df.sort_values('indexes')
df.drop(columns = ['min', 'max', 'sum', 'abs_sum', 'avg', 'std', 'count', 'nodata'], inplace=True)
print(df)
df.to_csv('alt_metric_stats.csv', index=False)

