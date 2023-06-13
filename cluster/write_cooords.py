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

base_dir = '/media/storage/glacier_dash_data/rgi/'
data = gp.read_file(base_dir + 'step2/rgi.shp').to_crs('EPSG:3857')

p = Proj('EPSG:3857')
x, y = p(data['CenLon'], data['CenLat'])
coords = np.c_[x, y]

from sklearn.cluster import KMeans
N = 24
clusters = KMeans(n_clusters=N, random_state=0, n_init="auto").fit(coords)
labels = clusters.labels_

xmin, ymin, xmax, ymax = data.total_bounds
#dx = xmax - xmin
#dy = ymax - ymin
#plt.xlim([xmin, xmax])
#plt.ylim([ymin, ymax])

data['cluster'] = labels
#print(data.groupby('cluster'))

#fig, ax = plt.subplots()
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


    print(len(np.unique(Z)))
    plt.imshow(Z)
    plt.colorbar()
    plt.show()
     
    output = rasterio.open(str(i) + 'tif', 'w', driver='GTiff',
                               height = ny, width = nx,
                               count=1, dtype=Z.dtype,
                               crs='EPSG:3857',
                               transform=transform,
                               nodata = 0,
                               compress='lzw')
   
    rect = patches.Rectangle((xmin, ymin), dx, dy, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    #print(i)
plt.show()


quit()
data.geometry = data.simplify(50)
poly = data.unary_union

def extract_poly_coords(geom):
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords += interior.coords[:]
    elif geom.type == 'MultiPolygon':
        exterior_coords = []
        interior_coords = []
        for part in geom:
            epc = extract_poly_coords(part)  # Recursive call
            exterior_coords += epc['exterior_coords']
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    return {'exterior_coords': exterior_coords,
            'interior_coords': interior_coords}

d = np.array(extract_poly_coords(poly)['exterior_coords'])
np.save('coords.npy', d)
