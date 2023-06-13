import geopandas as gp
import numpy as np
import pandas as pd
import functools
import json

"""
Merge stats from all datasets.
"""
datasets = []
decimals = [1, 1, 1, 2]
dataset_names = ['H_m', 'H_f', 'V_m', 'dhdt_h']
j = 0
for dataset_name in dataset_names:
    dataset = pd.read_csv(dataset_name + '_stats.csv', usecols=['indexes', 'avg', 'min', 'max'])
    dataset.rename(
        columns = {
            'avg' : dataset_name + '_avg',
            'min' : dataset_name + '_min',
            'max' : dataset_name + '_max',
        }, inplace = True)
    dataset.replace(-1e16, np.nan, inplace=True)
    dataset.replace(1e16, np.nan, inplace=True)
    dataset = dataset.round(decimals[j])
    print(dataset)
    datasets.append(dataset)
    print(dataset)
    j +=1

dataset = pd.read_csv('alt_metric_stats.csv')
dataset.rename(columns = {'metric' : 'dhdt_metric'}, inplace = True)
dataset = dataset.round(2)
datasets.append(dataset)
    
df = functools.reduce(lambda a, b : pd.merge(a,b,on='indexes',how='outer'), datasets)
#df['indexes'] = df['indexes'] 
df.rename(columns={'indexes' : 'index'}, inplace=True)


outlines = gp.read_file('outlines/outlines.shp')
df = pd.merge(df, outlines, on='index', how='outer')
df = gp.GeoDataFrame(df)
pd.DataFrame(df).to_csv('merged.csv')
#data = df.round(3)

print(df)
df.drop(columns=['Surging', 'Lmax', 'CenLon', 'CenLat'], inplace=True)
print(df.columns)
#quit()

data = json.loads(df.to_json(), parse_float=lambda x: round(float(x), 3))
data['name'] = 'glaciers'
data['crs'] =  { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } }

with open("all_data/outlines.json", "w") as outfile:
    outfile.write(json.dumps(data))
