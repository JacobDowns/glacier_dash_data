import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt

def plot_slope(i):
    print(i)
    try :
        store = xr.open_zarr('/media/storage/glacier_dash_data/netcdf_tiles/0.zarr/' + str(i))

        x = store['x'].data
        y = store['y'].data
        z = store['dem'].data

        # Get slope vector
        def get_slope(f):
            dx = abs(x[1] - x[0])
            dy = abs(y[1] - y[0])
            fx = np.gradient(f, dx, axis=1)
            fy = np.gradient(f, dy, axis=0)
            return fx, fy

        zx, zy = get_slope(z)
        slope = np.sqrt(zx**2 + zy**2)

        plt.imshow(np.rad2deg(np.arctan(slope)))
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(e)

plot_slope(13)
