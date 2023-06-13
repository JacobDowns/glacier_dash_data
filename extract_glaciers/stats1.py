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

file_names = glob.glob('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/*')
glacier_indexes = [int(f.split('/')[-1]) for f in file_names]
glacier_indexes = np.array(glacier_indexes)
glacier_indexes.sort()
#glacier_indexes = glacier_indexes[0:100]

def get_stats(i):
    print(i)
    try :
        store = xr.open_zarr('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/' + str(i))

        mask = store['mask'].data
        dhdt = store['dh'].data
        z = store['dem'].data
        store['v_live_2000'].data[store['v_live_2000'].data < 1e-3] = np.nan
        store['v_live_2017'].data[store['v_live_2017'].data < 1e-3] = np.nan
        store['v_live_err_2000'].data[store['v_live_err_2000'].data < 1e-3] = np.nan
        store['v_live_err_2017'].data[store['v_live_err_2017'].data < 1e-3] = np.nan

        indexes = mask > 0.
        area = (indexes.sum() * 50.*50.) / 1e6

        fields = {
            'dem' : 'z',
            'dh' : 'dh_h',
            'dh_err' : 'dh_h_err',
            'H_millan' : 'H_m',
            'H_err' : 'H_m_err',
            'H_farinotti' : 'H_f',
            'v' : 'v_m',
            'v_live_2000' : 'v_l_2000',
            'v_live_err_2000' : 'v_l_2000_err',
            'v_live_2017' : 'v_l_2017',
            'v_live_err_2017' : 'v_l_2017_err'
        }

        def get_vals(x):
            valid_indexes = ~np.isnan(x)
            return x[valid_indexes]


        d = {}
        d['index'] = i
        d['area'] = area.compute()
        for field in fields:
            x = store[field].data[indexes]
            x = get_vals(x)
            x = x.compute()

            name = fields[field]
            d[name + '_avg'] = np.mean(x) if len(x) > 0 else np.nan
            d[name + '_med'] = np.median(x) if len(x) > 0 else np.nan
            d[name + '_min'] = np.min(x) if len(x) > 0 else np.nan
            d[name + '_max'] = np.max(x) if len(x) > 0 else np.nan

        return d 
    except:
        print('err')

pool = Pool(processes=32)
results = pool.map(get_stats, glacier_indexes)
results1 = []
for r in results:
    if not r == None:
        results1.append(r)

        
df = pd.DataFrame(results1)
df.set_index('index')
df.sort_index()
df = df.round(3)

outlines = gp.read_file('/home/jake/glacier_dash_data/stats/outlines/outlines.shp')
outlines = outlines[['index', 'RGIId', 'Name']]
df = pd.merge(df, outlines, on='index', how='outer')
df.to_csv('stats.csv', index=False)
