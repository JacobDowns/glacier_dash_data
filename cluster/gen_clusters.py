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





coords = np.load('coords.npy')
N = 24
clusters = KMeans(n_clusters=N, random_state=0, n_init="auto").fit(coords)
#clusters =  DBSCAN(eps=0.3, min_samples=10).fit(data)
labels = clusters.labels_
colors = plt.get_cmap('jet')(np.linspace(0.,1.,N))
plt.scatter(coords[:,0], coords[:,1], c=colors[labels])
plt.show()

for i in range(N):
    indexes = labels == i
    coords_i = coords[indexes]
    plt.scatter(coords_i[:,0], coords_i[:,1])
    plt.show()
    
