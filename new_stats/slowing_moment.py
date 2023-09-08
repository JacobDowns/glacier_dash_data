import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xarray as xr
import zarr
from multiprocessing import Pool
import geopandas as gp

df = pd.read_csv('metrics.csv')
print(df)
quit()
data = pd.read_csv('stats_and_metric.csv', usecols = ['slow_m', 'thin_m', 'RGIId', 'Name'])
print(data)
data.to_csv('metrics.csv', index=False)

quit()

indexes = data['area'] > 15.
data = data[indexes]

plt.scatter(data['thin_m'], data['slow_m'], s=2)
plt.show()

plt.subplot(2,1,1)
plt.hist(data['thin_m'], bins=100)

plt.subplot(2,1,2)
plt.hist(data['slow_m'], bins=100)

plt.show()

quit()

df = pd.read_csv('stats.csv')
indexes = ~np.isnan(df['area'])
df = df[indexes]
glacier_indexes = df['index'].to_numpy()

def get_metric(i):
    print(i)
   
    store = xr.open_zarr('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/' + str(i))
    mask = store['mask'].data
    dhdt = store['dh'].data
    v_2000 = store['v_live_2000'].data
    v_2017 = store['v_live_2017'].data
    z = store['dem'].data

    indexes = np.logical_and(mask > 0., ~np.isnan(v_2000))
    indexes = np.logical_and(indexes, ~np.isnan(v_2017))
    
    z = z[indexes].compute()
    v_2000 = v_2000[indexes].compute()
    v_2017 = v_2017[indexes].compute()
    dhdt = dhdt[indexes].compute()

    d = {}
    d['index'] = i
    d['slow_m'] = np.nan
    d['thin_m'] = np.nan 
    
    if len(z) > 0:
        try :
            z_min = z.min()
            z_max = z.max()
            z_range = z.max() - z.min()

            def slow_moment(z, v_2000, v_2017):
                dv = v_2000 - v_2017
                dv[dv < 0.] = 0.
                dv += 1e-16
                moment_metric = ((dv * z) / dv.sum()).sum()
                moment_metric = (moment_metric - z_min) / (z_max - z_min)
                return moment_metric 


            def thin_moment(z, dhdt):
                dhdt[dhdt > 0.] = 0.
                dhdt += 1e-16
                moment_metric = ((dhdt * z) / dhdt.sum()).sum()
                moment_metric = (moment_metric - z_min) / (z_max - z_min)
                return moment_metric 

            d['thin_m'] = thin_moment(z, dhdt)
            d['slow_m'] = slow_moment(z, v_2000, v_2017)

        except Exception as e:
            print(e)
            print('err')
       

    return d


pool = Pool(processes=45)
results = pool.map(get_metric, glacier_indexes)
metric = pd.DataFrame(results)
df = pd.merge(df, metric, on='index', how='outer')
df = df.round(3)
df.to_csv('stats_and_metric.csv', index=False)
