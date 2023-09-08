import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gp
from multiprocessing import Pool

"""
Compute average and median thicknesses for all glaciers.
"""

# All rgi id's contained in the dataset
stats = pd.read_csv('stats_data.csv')
rgi_ids = list(stats['RGIId'])

# Function that gets average and median thickness for
# a given glacier
def get_stats(rgi_id):
    print(rgi_id)

    store = xr.open_zarr(f'glaciers.zarr/{rgi_id}')
    x = store['x'].data
    y = store['y'].data
    mask = store['mask'].data
    H = store['H_millan'].data
    H[mask == 0.] = np.nan

    d = {}
    d['RGIId'] = rgi_id
    d['H_avg'] = np.nanmean(H)
    d['H_med'] = np.nanmedian(H)

    return d

# This will compute stats for 4 glacier at once to
# speed things up
pool = Pool(processes=4)
results = pool.map(get_stats, rgi_ids)
# Convert results to a data frame
df = pd.DataFrame(results)
df = df.set_index('RGIId')
df = df.sort_values(by=['RGIId'])
df = df.round(3)
df.to_csv('stats_example.csv')
print(df)

