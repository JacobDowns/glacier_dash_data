import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
import geopandas as gp
from pyproj import Proj
import matplotlib.patches as patches
import rasterio
import math
from rasterio import features

"""
Script that clusters glaciers by proximity, then generates index raster 
tiles that contain glacier indexes at each pixel for each cluster.
"""

base_dir = '/media/storage/glacier_dash_data/rgi/'
data = gp.read_file(base_dir + 'step2/rgi.shp').to_crs('EPSG:3857')
print(data)

"""
Cluster by proximity.
"""
p = Proj('EPSG:3857')
x, y = p(data['CenLon'], data['CenLat'])
coords = np.c_[x, y]

from sklearn.cluster import KMeans
N = 24
clusters = KMeans(n_clusters=N, random_state=0, n_init="auto").fit(coords)
labels = clusters.labels_

xmin, ymin, xmax, ymax = data.total_bounds
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

"""
Generate index raster tiles.
"""
s = 0
data['cluster'] = labels
for i, d in data.groupby('cluster'):
    xmin, ymin, xmax, ymax = d.total_bounds
    res = 200.
    dx = xmax - xmin
    dy = ymax - ymin
    nx = math.ceil(dx / res)
    ny = math.ceil(dy / res)

    rgi_shapes = list(zip(d.geometry, d['index']))
    Z = np.zeros((ny, nx), dtype=int)
    transform = rasterio.Affine(200., 0, xmin, 0, -200., ymax)
    features.rasterize(shapes=rgi_shapes, out_shape=Z.shape, fill=0, out=Z, transform=transform)

    s += len(np.unique(Z))-1
    #plt.imshow(Z)
    #plt.colorbar()
    #plt.show()

    base_dir = '/media/storage/glacier_dash_data/index_tiles/'
    output = rasterio.open(base_dir + str(i) + '.tif', 'w', driver='GTiff',
                               height = ny, width = nx,
                               count=1, dtype=Z.dtype,
                               crs='EPSG:3857',
                               transform=transform,
                               nodata = 0,
                               compress='lzw')
    output.write(Z, 1)
    output.close()
   
    rect = patches.Rectangle((xmin, ymin), dx, dy, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    #print(i)
plt.show()
print(s)
