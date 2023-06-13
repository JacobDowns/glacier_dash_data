import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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
    z = store['dem'].data

    indexes = np.logical_and(mask > 0., ~np.isnan(dhdt))
    z = z[indexes].compute()
    dhdt = dhdt[indexes].compute()

    z_indexes = np.argsort(z)
    z = z[z_indexes]
    dhdt = dhdt[z_indexes]

    def moment_metric(z, dhdt):
        moment_metric = np.nan
        z = np.linspace(0.,1,len(z))
        return ((dhdt / dhdt.sum() + 1e-10) * z).sum()
        
    def my_metric(z, dhdt):

        z = (z - z.min()) / (z.max() - z.min())
        
        s, z1, z2 = binned_statistic(z, dhdt, bins=64)
        valid_indexes = ~np.isnan(s)
        s = s[valid_indexes]
        z = z1[:-1][valid_indexes]

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
        
    try:
        if len(z) > 0:
            z0, z1 = my_metric(z, dhdt)
            m = moment_metric(z, dhdt)
            
        d = {}
        d['index'] = i
        d['z_thick'] = z0 
        d['z_thin'] = z1
        d['metric'] = m
    except:
        d = {}
        d['index'] = i
        d['z_thick'] = np.nan 
        d['z_thin'] = np.nan
        d['metric'] = np.nan

    return d

pool = Pool(processes=45)
results = pool.map(get_metric, glacier_indexes)
metric = pd.DataFrame(results)
df = pd.merge(df, metric, on='index', how='outer')
df = df.round(3)
df.to_csv('stats_and_metric.csv', index=False)
