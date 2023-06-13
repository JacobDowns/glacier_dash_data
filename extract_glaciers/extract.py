import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd

#2600

def extract_clusters(tile_index):
    base_dir = '/media/jake/Backup/raster_data/'
    
    index_raster = rioxarray.open_rasterio(
        base_dir + 'index/{}.tif'.format(tile_index)
    )

    gindexes = np.unique(index_raster.data[0])[1:]

    rasters = {}

    rasters['index'] = rioxarray.open_rasterio(
        base_dir + 'index/{}.tif'.format(tile_index)
    )
    
    rasters['dem'] = rioxarray.open_rasterio(
        base_dir + 'dem/{}.tif'.format(tile_index)
    )
    
    rasters['H_millan'] = rioxarray.open_rasterio(
        base_dir + 'H_millan/{}.tif'.format(tile_index)
    )
    
    rasters['H_farinotti'] = rioxarray.open_rasterio(
        base_dir + 'H_farinotti/{}.tif'.format(tile_index)
    )
    
    rasters['dh'] = rioxarray.open_rasterio(
        base_dir + 'dhdt/{}.tif'.format(tile_index)
    )

    rasters['dh_err'] = rioxarray.open_rasterio(
        base_dir + 'dh_err/{}.tif'.format(tile_index)
    )

    rasters['v'] = rioxarray.open_rasterio(
        base_dir + 'V/{}.tif'.format(tile_index)
    )

    rasters['H_err'] = rioxarray.open_rasterio(
        base_dir + 'H_millan_err/{}.tif'.format(tile_index)
    )
    
    rasters['v'] = rioxarray.open_rasterio(
        base_dir + 'V/{}.tif'.format(tile_index)
    )
     
    rasters['vx'] = rioxarray.open_rasterio(
        base_dir + 'VX/{}.tif'.format(tile_index)
    )
     
    rasters['vy'] = rioxarray.open_rasterio(
        base_dir + 'VY/{}.tif'.format(tile_index)
    )


    for k in [0,17]:
        year = str(2000 + k)
        rasters['v_live_' + year] = rioxarray.open_rasterio(
            base_dir + 'live_velocity/v/{}/{}.tif'.format(year, tile_index)
        )

        rasters['v_live_err_' + year] = rioxarray.open_rasterio(
            base_dir + 'live_velocity/v_err/{}/{}.tif'.format(year, tile_index)
        ) 

    
    for index in gindexes:
        print('gindex', index)
        out_dir = '/media/storage/glacier_dash_data/glaciers/{}/'.format(index)

            
        indexes = np.argwhere(index_raster.data == index)

        xi = indexes[:,1]
        yi = indexes[:,2]

        xmin = xi.min()
        xmax = xi.max()
        ymin = yi.min()
        ymax = yi.max()
        xmin = max(0, xmin)
        xmax = min(xmax, rasters['H_millan'].y.size)
        ymin = max(0, ymin)
        ymax = min(ymax, rasters['H_millan'].x.size)

        mask = index_raster[0][xmin:xmax, ymin:ymax].copy(deep = True)
        mask.data[mask.data == index] = 1
        mask.data[~(mask.data == 1)] = 0

        #print(rasters['H_millan'].x.size)
        #print(rasters['H_millan'].y.size)

        #print(xmin, xmax)
        #print(mask.shape)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        mask.rio.to_raster(out_dir + 'mask.tif')
        mask.rio.to_raster(out_dir + 'mask.tif')

        R = rasters['H_millan'][0][xmin:xmax, ymin:ymax].copy(deep=True)
        H_avg = (R.data*mask.data).sum() / mask.data.sum()
        

        for k in rasters:
            #print(k)                
            R = rasters[k][0][xmin:xmax, ymin:ymax].copy(deep=True)
            R.rio.to_raster(out_dir + k + '.tif')
            #plt.imshow(R)
            #plt.colorbar()
            #plt.show()

for i in range(11):
    
    df = pd.read_csv('../stats/all_stats.csv')
    print(df)
    extract_clusters(11)
quit()

for x, y in df.groupby('tile', as_index=False):
    extract_clusters(x, y['indexes'])

