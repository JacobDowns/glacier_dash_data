from dem_stitcher.stitcher import stitch_dem
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
from pathlib import Path

dst_area_or_point = 'Point'
dst_ellipsoidal_height = True
dem_name = 'glo_30'
out_directory_name = 'out'

out_directory = Path(out_directory_name)
out_directory.mkdir(exist_ok=True, parents=True)

# Central coast of California
# xmin, ymin, xmax, ymax
bounds = [-121.5, 34.95, -120.2, 36.25]

X, p = stitch_dem(bounds,
                  dem_name=dem_name,
                  dst_ellipsoidal_height=dst_ellipsoidal_height,
                  dst_area_or_point=dst_area_or_point)

print(X.shape)
fig, ax = plt.subplots(figsize=(8, 8))
ax = plot.show(X, transform=p['transform'], ax=ax)
ax.set_xlabel('Longitude', size=15)
ax.set_ylabel('Latitude', size=15)

plt.show()
