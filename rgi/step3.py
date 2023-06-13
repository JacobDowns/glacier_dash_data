import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
import geopandas as gp
from sklearn.cluster import KMeans
#from sklearn.cluster import Ward
from sklearn.cluster import DBSCAN 
data = np.load('d.npy')
print(data)
N = 24
clusters = KMeans(n_clusters=N, random_state=0, n_init="auto").fit(data)
#clusters =  DBSCAN(eps=0.3, min_samples=10).fit(data)
print(clusters.labels_)
print(np.unique(clusters.labels_))
#quit()
colors = plt.get_cmap('jet')(np.linspace(0.,1.,N))
plt.scatter(data[:,0], data[:,1], c=colors[clusters.labels_])
plt.show()
quit()

plt.plot(data[:,0], data[:,1], 'ko')
plt.show()
quit()

base_dir = '/media/storage/glacier_dash_data/rgi/'
data = gp.read_file(base_dir + 'step2/rgi.shp').to_crs('EPSG:3857')
#data = data[0:10]
#print(data)
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
np.save('d.npy', d)

quit()


#data.plot()
#plt.show()

#coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]
def coord_lister(geom):
    coords = list(geom.exterior.coords)
    return (coords)

coordinates_list = data.geometry.apply(coord_lister)
print(coordinates_list)
