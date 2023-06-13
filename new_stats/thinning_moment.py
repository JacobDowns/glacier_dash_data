import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xarray as xr
import zarr
from multiprocessing import Pool
import geopandas as gp

#data = pd.read_csv('stats_and_metric.csv')
#print(data)
#quit()

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
    h = store['H_millan'].data

    indexes = np.logical_and(mask > 0., ~np.isnan(dhdt))
    indexes = np.logical_and(indexes, ~np.isnan(dhdt_err))
    
    z = z[indexes].compute()
    dhdt = dhdt[indexes].compute()
    dhdt_err = dhdt_err[indexes].compute()
    h = h[indexes].compute()
    
    z_indexes = np.argsort(z)
    z = z[z_indexes]
    dhdt = dhdt[z_indexes]
    dhdt_err = dhdt_err[z_indexes]
    h = h[z_indexes]

    d = {}
    d['index'] = i
    d['m0'] = np.nan 
    d['m1'] = np.nan

    
    if len(z) > 0:
        try :
            z_min = z.min()
            z_max = z.max()
            z_range = z.max() - z.min()

            #def time_metric(h, dhdt):
            #    return abs(h.mean()) / abs(dhdt.mean() + 1e-10)

            def moment_metric(z, dhdt, dhdt_err):
                z = np.linspace(0.,1,len(z))
                w = dhdt / (dhdt_err + 1e-10)
                return (w * z / (w.sum() + 1e-10)).sum()

            def thinning_moment(z, dhdt):
                dhdt[dhdt > 0.] = 0.
                dhdt += 1e-16
                moment_metric = ((dhdt * z) / dhdt.sum()).sum()
                moment_metric = (moment_metric - z_min) / (z_max - z_min)
                return moment_metric 

            def thickening_moment(z, dhdt):
                dhdt[dhdt < 0.] = 0.
                dhdt += 1e-16
                moment_metric = ((dhdt * z) / dhdt.sum()).sum()
                moment_metric = (moment_metric - z_min) / (z_max - z_min)
                return moment_metric 

            d['m0'] = thinning_moment(z, dhdt)
            d['m1'] = thickening_moment(z, dhdt)


        except:
            print('err')
       

    return d


pool = Pool(processes=45)
results = pool.map(get_metric, glacier_indexes)
metric = pd.DataFrame(results)
df = pd.merge(df, metric, on='index', how='outer')
df = df.round(3)
df.to_csv('stats_and_metric.csv', index=False)
