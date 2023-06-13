import numpy as np
import rioxarray
import xarray
import matplotlib.pyplot as plt
import glob
import os
import sys
import pandas as pd
import geopandas as gp
import json



base_dir = '/media/storage/glacier_dash_data/rgi/'



data = gp.read_file(base_dir + 'step2/rgi.shp')

data = data.to_crs('EPSG:3857')
#data.geometry = data.simplify(50)

data = data.to_crs('EPSG:4326')
#indexes = data['Area'] > .25
#data = data[indexes]
data = data.drop(columns=['GLIMSId', 'BgnDate', 'EndDate', 'O1Region', 'O2Region', 'Status', 'Connect', 'Form', 'TermType', 'Linkages'])

data.to_file('outlines/outlines.shp', index=False)
