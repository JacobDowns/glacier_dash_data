import numpy as np
import pandas as pd
import geopandas as gp
import json

df = pd.read_csv('stats.csv')
print(df.columns)
df['Name'] = df['Name'].replace(np.nan, 'None')


df['H_m_avg'] = df['H_m_avg'].replace(0., 'null')
df['H_m_max'] = df['H_m_max'].replace(0., 'null')
df['H_m_med'] = df['H_m_med'].replace(0., 'null')

df['H_f_avg'] = df['H_f_avg'].replace(0., 'null')
df['H_f_max'] = df['H_f_max'].replace(0., 'null')
df['H_f_med'] = df['H_f_med'].replace(0., 'null')


outlines = gp.read_file('/home/jake/glacier_dash_data/stats/outlines/outlines.shp')
outlines = outlines[['index', 'geometry', 'CenLon', 'CenLat', 'Area', 'Slope', 'Aspect']]
df = pd.merge(df, outlines, on='index', how='outer')
df = gp.GeoDataFrame(df)
data = json.loads(df.to_json())
data['name'] = 'glaciers'
data['crs'] =  { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } }

with open("all_data/outlines.json", "w") as outfile:
    outfile.write(json.dumps(data))
