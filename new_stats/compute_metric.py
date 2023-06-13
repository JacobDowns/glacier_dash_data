import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xarray as xr
import zarr
from scipy.stats import binned_statistic
from multiprocessing import Pool
import geopandas as gp
from scipy.stats import binned_statistic

df = pd.read_csv('stats.csv')
indexes = ~np.isnan(df['area'])
df = df[indexes]
glacier_indexes = df['index'].to_numpy()

def get_metric(i):
    print(i)
   
    store = xr.open_zarr('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/' + str(i))
    mask = store['mask'].data
    dhdt = store['dh'].data
    dhdt_err = store['dh_err'].data
    z = store['dem'].data

    indexes = np.logical_and(mask > 0., ~np.isnan(dhdt))
    indexes = np.logical_and(indexes, ~np.isnan(dhdt_err))
    z = z[indexes].compute()
    dhdt = dhdt[indexes].compute()
    dhdt_err = dhdt_err[indexes].compute()

    z_indexes = np.argsort(z)
    z = z[z_indexes]
    dhdt = dhdt[z_indexes]
    dhdt_err = dhdt_err[z_indexes]


    def moment_metric(z, dhdt, dhdt_err):
        moment_metric = np.nan
        z = np.linspace(0.,1,len(z))
        w = dhdt / (dhdt_err + 1e-10)
        return (w * z / (w.sum() + 1e-10)).sum()
        
    def my_metric(z, dhdt):

        z = (z - z.min()) / (z.max() - z.min())
        
        dh, z1, z2 = binned_statistic(z, dhdt, bins=64)
        dh_err, z1, z2 = binned_statistic(z, dhdt_err, bins=64)
        z = z1[:-1]

        dh[np.isnan(dh)] = 0.
        dh_err[np.isnan(dh_err)] = 100.

        

        plt.subplot(2,1,1)
        plt.plot(z, dh)

        plt.subplot(2,1,1)
        plt.plot(z, dh_err)

        
        plt.show()

        
        

        s0 = np.zeros_like(s)
        s1 = np.zeros_like(s)
        s0[:] = s[:]
        s1[:] = s[:]
        s0[s < 0.] = 0.
        s1[s > 0.] = 0.
        s1 = abs(s1)
        
        s0 = np.cumsum(s0) / (s0.sum() + 1e-16)
        z0 = z[np.argwhere(s0 < 0.5).flatten()[-1]]

        s1 = np.cumsum(s1) / (s1.sum() + 1e-16)
        z1 = z[np.argwhere(s1 < 0.5).flatten()[-1]]

        return z0, z1


    d = {}
    d['index'] = i
    d['z_thick'] = np.nan
    d['z_thin'] = np.nan
    d['metric'] = np.nan
        
    try:
        if len(z) > 0:
            m = moment_metric(z, dhdt, dhdt_err)
            z0, z1 = my_metric(z, dhdt)
            
            print(m)
            d = {}
            d['index'] = i
            d['z_thick'] = z0 
            d['z_thin'] = z1
            d['metric'] = m
    except:
       print('err')

    return d


for i in range(1,5):
    get_metric(i)

quit()
pool = Pool(processes=45)
results = pool.map(get_metric, glacier_indexes)
metric = pd.DataFrame(results)
df = pd.merge(df, metric, on='index', how='outer')
df = df.round(3)
df.to_csv('stats_and_metric.csv', index=False)
