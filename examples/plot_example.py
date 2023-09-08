import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

rgi_id = 'RGI60-01.09162'
store = xr.open_zarr(f'glaciers.zarr/{rgi_id}')
x = store['x'].data
y = store['y'].data
z = store['dem'].data
mask = store['mask'].data
H = store['H_millan'].data
dhdt = store['dhdt_2000-2020'].data
v_mean = store['V_2000-2019'].data

H[mask == 0.] = np.nan
dhdt[mask == 0.] = np.nan
v_mean[mask == 0.] = np.nan

plt.subplot(2,2,1)
plt.title('Thickness (Millan)')
plt.imshow(H)
plt.colorbar()

plt.subplot(2,2,2)
plt.title('DEM (Copernicus)')
plt.imshow(z)
plt.colorbar()

plt.subplot(2,2,3)
plt.title(r'$\frac{dH}{dt}$ (Hugonnet)')
plt.imshow(dhdt)
plt.colorbar()

plt.subplot(2,2,4)
plt.title(r'Mean velocity 2000-2019 (ITS_LIVE)')
plt.imshow(v_mean)
plt.colorbar()

plt.show()

