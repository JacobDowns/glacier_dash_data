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
df.set_index('index')

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


    def moment_metric0(z, dhdt, dhdt_err):
        z = np.linspace(0.,1,len(z))
        w = dhdt / (dhdt_err + 1e-10)
        return (w * z / (w.sum() + 1e-10)).sum()

    def moment_metric1(z, dhdt, dhdt_err):
        z = np.linspace(0.,1,len(z))
        w = dhdt
        return (w * z / (w.sum() + 1e-10)).sum()
        
    def my_metric(z, dhdt, dhdt_err):

        z_min = z.min()
        z_max = z.max()
        z_range = z.max() - z.min()
        z = (z - z.min()) / z_range
        
        dh, z1, z2 = binned_statistic(z, dhdt, bins=64)
        dh_err, z1, z2 = binned_statistic(z, dhdt_err, bins=64)
        z = z1[:-1]

        dh[np.isnan(dh)] = 0.
        dh_err[np.isnan(dh_err)] = 100.

        coeffs = np.polyfit(z, dh, 1, w=1./(dh_err + 1e-10))
        a, b  = coeffs

        plt.fill_between(z, dh-np.sqrt(dh_err), dh+np.sqrt(dh_err), alpha=0.2)
        plt.plot(z, dh)
        plt.plot(z, a*z + b, 'ro-')
        plt.show()

        """
        coeffs = np.polyfit(z, dh, 3, w=1./(dh_err + 1e-10))
        a, b, c, d = coeffs
        
        

        plt.fill_between(z, dh-np.sqrt(dh_err), dh+np.sqrt(dh_err), alpha=0.2)
        plt.plot(z, dh)
        plt.plot(z, a*z**3 + b*z**2 + c*z + d, 'ro-')
        plt.show()
        """
        slope = (a*z_range) / 1e3
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
        plt.plot(z, dh1)
        plt.show()
        z1 = z[np.argwhere(dh1 < 0.5).flatten()[-1]]

        return slope, z0, z1


    d = {}
    d['index'] = i
    d['dhdt_slope'] = np.nan
    d['z_thick'] = np.nan
    d['z_thin'] = np.nan
    d['metric0'] = np.nan
    d['metric1'] = np.nan

    
   
    if len(z) > 0:
        m0 = moment_metric0(z, dhdt, dhdt_err)
        m1 = moment_metric0(z, dhdt, dhdt_err)
        d['metric0'] = m0
        d['metric1'] = m1


        dhdt_slope, z0, z1 = my_metric(z, dhdt, dhdt_err)
        d['dhdt_slope'] = dhdt_slope
        d['z_thick'] = z0 
        d['z_thin'] = z1

        print(d)
        print(df[df['index'] == i]['Name'])

        return d

get_metric(10685)
get_metric(10553)
get_metric(13692)

