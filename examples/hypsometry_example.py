import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Basic plot of hypsometry of Wolverine glacier
rgi_id = 'RGI60-01.09162'
store = xr.open_zarr(f'glaciers.zarr/{rgi_id}')
x = store['x'].data
y = store['y'].data
z = store['dem'].data
mask = store['mask'].data

# Get pixels on glacier
indexes = mask == 1.
z = z[indexes]

# Plot Area v. elevation distribution
bins = np.linspace(z.min(), z.max(), 25)
plt.hist(z, bins, density=True)
plt.show()

