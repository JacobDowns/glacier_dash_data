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

data = pd.read_csv('stats_and_metric.csv')


plt.hist(data['metric'], bins=np.linspace(-1.5,1.5))
plt.show()

plt.hist(data['dhdt_slope'], bins=np.linspace(-1.5,1.5, 100))
plt.show()

print(data)
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
        
    def my_metric(z, dhdt, dhdt_err):

        z_range = z.max() - z.min()
        z = (z - z.min()) / z_range
        
        dh, z1, z2 = binned_statistic(z, dhdt, bins=64)
        dh_err, z1, z2 = binned_statistic(z, dhdt_err, bins=64)
        z = z1[:-1]

        dh[np.isnan(dh)] = 0.
        dh_err[np.isnan(dh_err)] = 100.
        
        coeffs = np.polyfit(z, dh, 1, w=1./(dh_err + 1e-10))
        m, b = coeffs
        slope = (m*z_range) / 1e3
        

        dh0 = np.zeros_like(dh)
        dh1 = np.zeros_like(dh)
        dh0[:] = dh[:]
        dh1[:] = dh[:]
        dh0[dh < 0.] = 0.
        dh1[dh > 0.] = 0.
        dh1 = abs(dh1)
        
        dh0 = np.cumsum(dh0) / (dh0.sum() + 1e-10)
        z0 = z[np.argwhere(dh0 < 0.5).flatten()[-1]]

        dh1 = np.cumsum(dh1) / (dh1.sum() + 1e-16)
        z1 = z[np.argwhere(dh1 < 0.5).flatten()[-1]]

        return slope, z0, z1


    d = {}
    d['index'] = i
    d['dhdt_slope'] = np.nan
    d['z_thick'] = np.nan
    d['z_thin'] = np.nan
    d['metric'] = np.nan

    try:
        if len(z) > 0:
            m = moment_metric(z, dhdt, dhdt_err)
            d['metric'] = m
        
            dhdt_slope, z0, z1 = my_metric(z, dhdt, dhdt_err)
            d['dhdt_slope'] = dhdt_slope
            d['z_thick'] = z0 
            d['z_thin'] = z1
    except:
        print('err')

    return d


#for i in range(1,5):
#    get_metric(i)
#quit()

pool = Pool(processes=45)
results = pool.map(get_metric, glacier_indexes)
metric = pd.DataFrame(results)
df = pd.merge(df, metric, on='index', how='outer')
df = df.round(3)
df.to_csv('stats_and_metric.csv', index=False)
