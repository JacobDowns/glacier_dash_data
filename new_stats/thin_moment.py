import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xarray as xr
import zarr
from multiprocessing import Pool
import geopandas as gp


data = pd.read_csv('stats_and_metric.csv', usecols = ['slow_m', 'thin_m', 'RGIId', 'Name'])

#indexes = data['area'] > 5.
#data = data[indexes]

plt.scatter(data['thin_m'], data['slow_m'], s=2)
plt.show()

plt.subplot(2,1,1)
plt.hist(data['thin_m'], bins=100)

plt.subplot(2,1,2)
plt.hist(data['slow_m'], bins=100)

plt.show()

df.to_csv('metrics.csv', index=False)
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
    v_2000 = store['v_live_2000'].data
    v_2017 = store['v_live_2017'].data
    eps = -1e-16
    
    d = {}
    d['index'] = i
    d['slow_m'] = np.nan
    d['thin_m'] = np.nan

    def thin_moment(z, dhdt):
        indexes = np.logical_and(mask > 0., ~np.isnan(dhdt))
        z = z[indexes].compute()
        dhdt = dhdt[indexes].compute()
        z_indexes = np.argsort(z)
        z = z[z_indexes]
        dhdt = dhdt[z_indexes]

        if len(z) > 0:
            if dhdt.min() >= 0.:
                return np.nan
            
            dhdt[dhdt > eps] = eps
            z_n = np.linspace(0.,1,len(z))
            w = dhdt / dhdt.sum()
            return (w*z_n).sum()
        else :
            return np.nan

    def slow_moment(z, v_2000, v_2017):
        indexes = np.logical_and(mask > 0., ~np.isnan(v_2000))
        indexes = np.logical_and(indexes, ~np.isnan(v_2017))

        z = z[indexes].compute()
        v_2017 = v_2017[indexes].compute()
        v_2000 = v_2000[indexes].compute()

        z_indexes = np.argsort(z)
        z = z[z_indexes]
        v_2000 = v_2000[z_indexes]
        v_2017 = v_2017[z_indexes]
        dv = v_2000 - v_2017
        
        if len(z) > 0:
            if dv.min() >= 0.:
                return np.nan
            
            dv[dv > eps] = eps
            z_n = np.linspace(0.,1,len(z))
            w = dv / dv.sum()
            return (w*z_n).sum()
        else :
            return np.nan


    try:
        d['thin_m'] = thin_moment(z, dhdt)
        d['slow_m'] = slow_moment(z, v_2000, v_2017)
    except Exception as e:
        print(e)
   

    return d


#for i in range(1,5):
#    print(get_metric(i))
    



pool = Pool(processes=45)
results = pool.map(get_metric, glacier_indexes)
metric = pd.DataFrame(results)
df = pd.merge(df, metric, on='index', how='outer')
df = df.round(3)
df.to_csv('stats_and_metric.csv', index=False)

print(df)
