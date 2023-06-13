import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
import zarr
from datatree import DataTree, open_datatree
from scipy.stats import binned_statistic
from multiprocessing import Pool
import geopandas as gp
from scipy.stats import binned_statistic


df = pd.read_csv('stats_and_metric.csv')
print(df)

plt.hist(df['metric1'], bins=np.linspace(-2.,2.,100))
plt.show()

plt.hist(df['metric2'], bins=np.linspace(0.,1.,50))

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
    z = store['dem'].data



    indexes = np.logical_and(mask > 0., ~np.isnan(dhdt))
    z = z[indexes].compute()
    dhdt = dhdt[indexes].compute()

    z_indexes = np.argsort(z)
    z = z[z_indexes]
    dhdt = dhdt[z_indexes]

    metric1 = np.nan
    metric2 = np.nan

    try:
        if len(z) > 0:
            z = (z - z.min()) / (z.max() - z.min())

            s, z1, z2 = binned_statistic(z, dhdt, bins=32)
            valid_indexes = ~np.isnan(s)
            s = s[valid_indexes]
            z1 = z1[:-1][valid_indexes]
            s[s > 0.] = 0.

            s = np.cumsum(s) / s.sum()
            zi = z1[np.argwhere(s < 0.5).flatten()[-1]]
            #print(zi)
            
            #plt.plot(z1, np.cumsum(s) / s.sum())
            #plt.show()
            
            metric2 = zi
            metric1 = ((dhdt / dhdt.sum() + 1e-10) * z).sum()
            #metric2 = ( (dhdt - dhdt.max()) /(dhdt - dhdt.max() +1e-10).sum() * z).sum()

        d = {}
        d['index'] = i
        d['metric1'] = metric1
        d['metric2'] = metric2
    except:
        d = {}
        d['metric1'] = np.nan
        d['metric2'] = np.nan
   
    return d

#for i in range(1,5):
#    get_metric(i)
#quit()

pool = Pool(processes=40)
results = pool.map(get_metric, glacier_indexes)
metric = pd.DataFrame(results)
df = pd.merge(df, metric, on='index', how='outer')
df.to_csv('stats_and_metric.csv', index=False)
