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

    z0 = np.zeros_like(z)
    z1 = np.zeros_like(z)
    z0[:] = z[:]
    t = 1000.
    z1[:] = z[:] - t*dhdt.mean()
    
    z0_min = z0.min()
    z0_max = z0.max()
    z1_min = z1.min()
    z1_max = z1.max()


    indexes = np.argsort(dhdt)
    dhdt = dhdt[indexes]
    z = z[indexes]
    a = np.linspace(0.,1.,len(z))

    
    
    plt.plot(dhdt, a)
    plt.show()

    
    
    quit()
    

    
    #z0 = (z0 - z0_min) / (z0_max - z0_min)
    #z1 = (z1 - z0_min) / (z0_max - z0_min)


    
    x = np.linspace(0.,1.,len(z0))

    plt.plot( z0[1:] / (z0[1:] - z0[:-1]), 'ko')
    plt.show()
    quit()

    #plt.plot(x[:-1], z0[1:] -z1[:-1])
    #plt.show()
    #quit()
    
    #print(10.*dhdt.mean())
    print(np.mean(x*z0))
    print(np.mean(x*z1))
    print(np.mean(z0-z1))

    
    #z1.sort()
    
    #bins = np.linspace(0.,1.,100)
    print((z0 - z1).mean())
    print(dhdt.mean())
    print(z0.mean() - z1.mean())


    plt.plot(z0, x, 'r')
    plt.plot(z1, x, 'b')
    plt.show()

    
   
    

    


#get_metric(11)
#quit()
#get_metric(10685)
#get_metric(10553)
get_metric(13692)

