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
from scipy.interpolate import interp1d

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

    z0 = np.zeros_like(z)
    z1 = np.zeros_like(z)
    z0[:] = z[:]
    t = 19.
    z1[:] = z[:] + t*dhdt
    z1.sort()
    
    z0_min = z0.min()
    z0_max = z0.max()
    z1_min = z1.min()
    z1_max = z1.max()    
    x = np.linspace(0.,1.,len(z0))
    
    a0 = interp1d(z0, x)
    a1 = interp1d(z1, x)

    zs0 = np.linspace(z0_min, z0_max, 1000)
    zs1 = np.linspace(z1_min, z1_max, 1000)
    a0 = a0(zs0)
    a1 = a1(zs1)

    plt.plot(zs0, a0, label = r'$a_0(z)$', linewidth=2)
    plt.plot(zs1, a1, label = r'$a_1(z)$', linewidth=2)
    plt.legend()
    plt.show()

    w0 = (a0[1:] - a0[:-1])
    w0[np.isnan(w0)] = 0.
    w0 /= w0.sum()
    
    plt.plot(zs0[1:], w0)

    w1 = (a1[1:] - a1[:-1])
    w1[np.isnan(w1)] = 0.
    w1 /= w1.sum()

    plt.plot(zs1[1:], w1)
    plt.show()

    print(t*dhdt.mean())
    print((w0*zs0[1:]).sum())
    print((w1*zs1[1:]).sum())
    print((w1*zs1[1:]).sum() - (w0*zs0[1:]).sum())


get_metric(11)
#quit()
get_metric(10685)
get_metric(10553)
get_metric(13692)

