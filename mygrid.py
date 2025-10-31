
'''
import os
import numpy as np
import pyroms
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from bathy_smoother import bathy_tools, bathy_smoothing

# -----------------------------
# 1. Grid resolution and domain
# -----------------------------
res = 1  # 1/12 degree resolution
lonp = np.array([30., 30., 120., 120.])   # lon0, lon1, lon2, lon3
latp = np.array([30., -30., -30., 30.])   # lat0, lat1, lat2, lat3
beta = np.array([1, 1, 1, 1])             # boundary type

# Number of grid points (without ghost points)
Lm = int((120.-30.) / res)   # 1080
Mm = int((30.-(-30.)) / res) # 720

# -----------------------------
# 2. Basemap (Plate Carrée projection)
# -----------------------------
m = Basemap(projection='cyl', llcrnrlon=30, urcrnrlon=120,
            llcrnrlat=-30, urcrnrlat=30, resolution='i')

m.drawcoastlines()
m.drawparallels(np.arange(-30, 31, 10))
m.drawmeridians(np.arange(30, 121, 10))

# -----------------------------
# 3. Boundary Interactor
# -----------------------------
xp, yp = m(lonp, latp)
bry = pyroms.hgrid.BoundaryInteractor(xp, yp, beta, shp=(Mm+3, Lm+3), proj=m)

# Generate 2D x/y arrays
bx = np.linspace(xp.min(), xp.max(), Lm+3)
by = np.linspace(yp.min(), yp.max(), Mm+3)
bx2d, by2d = np.meshgrid(bx, by)

# Horizontal grid
hgrd = pyroms.hgrid.CGrid_geo(bx2d, by2d, m)

# -----------------------------
# 4. Load ETOPO bathymetry
# -----------------------------
datadir = 'data/'  # Update to your local path
topo = np.loadtxt(os.path.join(datadir, 'etopo20data.gz'))
lons = np.loadtxt(os.path.join(datadir, 'etopo20lons.gz'))
lats = np.loadtxt(os.path.join(datadir, 'etopo20lats.gz'))

topo = -topo  # depth positive
hmin = 5
topo = np.where(topo < hmin, hmin, topo)

# Interpolate bathymetry onto ROMS grid
lon, lat = np.meshgrid(lons, lats)
h = griddata((lon.flatten(), lat.flatten()), topo.flatten(),
             (hgrd.lon_rho, hgrd.lat_rho), method='linear')
h = np.where(h < hmin, hmin, h)

# -----------------------------
# 5. Mask land using Basemap
# -----------------------------
for xx, yy in m.coastpolygons:
    vv = np.zeros((len(xx), 2))
    vv[:, 0] = xx
    vv[:, 1] = yy
    hgrd.mask_polygon(vv, mask_value=0)

# Set depth to hmin where masked
idx = np.where(hgrd.mask_rho == 0)
h[idx] = hmin

# Save raw bathymetry for vertical grid
hraw = h.copy()

# -----------------------------
# 6. Smooth bathymetry
# -----------------------------
rx0_max = 0.35
h = bathy_smoothing.smoothing_Positive_rx0(hgrd.mask_rho, h, rx0_max)

# Assign smoothed depth to horizontal grid
hgrd.h = h

# -----------------------------
# 7. Vertical grid (s-coordinate)
# -----------------------------
theta_b = 2
theta_s = 7.0
Tcline = 50
N = 30
vgrd = pyroms.vgrid.s_coordinate_4(hgrd.h, theta_b, theta_s, Tcline, N, hraw=hraw)

# -----------------------------
# 8. Create ROMS grid
# -----------------------------
grd_name = 'INDO_1_12deg'
grd = pyroms.grid.ROMS_Grid(grd_name, hgrd, vgrd)

# -----------------------------
# 9. Write ROMS grid to NetCDF
# -----------------------------
pyroms.grid.write_ROMS_grid(grd, filename='INDO_1_12deg_grd.nc')

print("Grid successfully written to INDO_1_12deg_grd.nc")
'''
import os
import numpy as np
import pyroms
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from bathy_smoother import bathy_tools, bathy_smoothing

# -----------------------------
# 1. Grid resolution and domain
# -----------------------------
res = 1/12  # 1/12 degree resolution
lonp = np.array([30., 30., 120., 120.])   # lon0, lon1, lon2, lon3
latp = np.array([30., -30., -30., 30.])   # lat0, lat1, lat2, lat3
beta = np.array([1, 1, 1, 1])             # boundary type

# Number of grid points (without ghost points)
Lm = int((120.-30.) / res)   # 1080
Mm = int((30.-(-30.)) / res) # 720

# -----------------------------
# 2. Basemap (Plate Carrée projection)
# -----------------------------
m = Basemap(projection='cyl', llcrnrlon=30, urcrnrlon=120,
            llcrnrlat=-30, urcrnrlat=30, resolution='i')

m.drawcoastlines()
m.drawparallels(np.arange(-30, 31, 10))
m.drawmeridians(np.arange(30, 121, 10))

# -----------------------------
# 3. Boundary Interactor
# -----------------------------
xp, yp = m(lonp, latp)
bry = pyroms.hgrid.BoundaryInteractor(xp, yp, beta, shp=(Mm+3, Lm+3), proj=m)

# Generate 2D x/y arrays
bx = np.linspace(xp.min(), xp.max(), Lm+3)
by = np.linspace(yp.min(), yp.max(), Mm+3)
bx2d, by2d = np.meshgrid(bx, by)

# Horizontal grid
hgrd = pyroms.hgrid.CGrid_geo(bx2d, by2d, m)

# -----------------------------
# 4. Load ETOPO bathymetry
# -----------------------------
datadir = 'data/'  # Update to your local path
topo = np.loadtxt(os.path.join(datadir, 'etopo20data.gz'))
lons = np.loadtxt(os.path.join(datadir, 'etopo20lons.gz'))
lats = np.loadtxt(os.path.join(datadir, 'etopo20lats.gz'))

topo = -topo  # depth positive
hmin = 5
topo = np.where(topo < hmin, hmin, topo)

# Interpolate bathymetry onto ROMS grid
lon, lat = np.meshgrid(lons, lats)
h = griddata((lon.flatten(), lat.flatten()), topo.flatten(),
             (hgrd.lon_rho, hgrd.lat_rho), method='linear')
h = np.where(h < hmin, hmin, h)

# -----------------------------
# 5. Mask land using Basemap
# -----------------------------
for xx, yy in m.coastpolygons:
    vv = np.zeros((len(xx), 2))
    vv[:, 0] = xx
    vv[:, 1] = yy
    hgrd.mask_polygon(vv, mask_value=0)

# Set depth to hmin where masked
idx = np.where(hgrd.mask_rho == 0)
h[idx] = hmin

# Save raw bathymetry for vertical grid
hraw = h.copy()

# -----------------------------
# 6. Smooth bathymetry
# -----------------------------
rx0_max = 0.35
h = bathy_smoothing.smoothing_Positive_rx0(hgrd.mask_rho, h, rx0_max)

# Assign smoothed depth to horizontal grid
hgrd.h = h

# -----------------------------
# 7. Vertical grid (s-coordinate)
# -----------------------------
theta_b = 0.1
theta_s = 7.0
Tcline = 250
N = 40
vgrd = pyroms.vgrid.s_coordinate_2(hgrd.h, theta_b, theta_s, Tcline, N,hraw=hraw)

# -----------------------------
# 8. Create ROMS grid
# -----------------------------
grd_name = 'INDO_1_12deg'
grd = pyroms.grid.ROMS_Grid(grd_name, hgrd, vgrd)

# -----------------------------
# 9. Write ROMS grid to NetCDF
# -----------------------------
#pyroms.grid.write_ROMS_grid(grd, filename='INDO_1_12deg_grd.nc')

#print("Grid successfully written to INDO_1_12deg_grd.nc")


# In[41]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RectangleSelector

mask_edit = hgrd.mask_rho.copy()

# Define discrete colormap: 0 = land (sandy brown), 1 = ocean (light blue)
cmap = ListedColormap(['sandybrown', 'lightblue'])

fig, ax = plt.subplots(figsize=(12, 8))
mesh = ax.imshow(mask_edit, origin='lower', cmap=cmap, interpolation='none')

# Add gridlines
ax.set_xticks(np.arange(-0.5, mask_edit.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, mask_edit.shape[0], 1), minor=True)
ax.grid(which='minor', color='gray', linewidth=0.1)
ax.set_title("Click to toggle a cell OR drag to toggle a region.\nClose figure when done.")

def onclick(event):
        if event.xdata is None or event.ydata is None:
                    return
        j = int(event.xdata)
        i = int(event.ydata)
        if i < 0 or i >= mask_edit.shape[0] or j < 0 or j >= mask_edit.shape[1]:
                return
                                        # Toggle single cell
        mask_edit[i, j] = 1 - mask_edit[i, j]
        mesh.set_data(mask_edit)
        fig.canvas.draw_idle()



def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

                # Ensure proper bounds
        i_min, i_max = sorted([int(y1), int(y2)])
        j_min, j_max = sorted([int(x1), int(x2)])

        i_min = max(i_min, 0)
        j_min = max(j_min, 0)
        i_max = min(i_max, mask_edit.shape[0] - 1)
        j_max = min(j_max, mask_edit.shape[1] - 1)

                                            # Determine majority state in region to decide toggle direction
        region = mask_edit[i_min:i_max + 1, j_min:j_max + 1]
        mean_val = np.mean(region)
        new_val = 0 if mean_val > 0.5 else 1  # Toggle to opposite type (land↔ocean)

        mask_edit[i_min:i_max + 1, j_min:j_max + 1] = new_val
        mesh.set_data(mask_edit)
        fig.canvas.draw_idle()







cid = fig.canvas.mpl_connect('button_press_event', onclick)

toggle_selector = RectangleSelector(ax, onselect,
        drawtype='box',useblit=True,button=[3],  # Right-click for box select
        interactive=True,spancoords='pixels')

plt.show()


hgrd.mask_rho = mask_edit
h[hgrd.mask_rho == 0] = hmin
pyroms.grid.write_ROMS_grid(grd, filename='IO_1_12deg_grid.nc')

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Copy the ROMS mask for editing
mask_edit = hgrd.mask_rho.copy()

# Define discrete colormap: 0 = land (sandy brown), 1 = ocean (light blue)
cmap = ListedColormap(['sandybrown', 'lightblue'])

fig, ax = plt.subplots(figsize=(12, 8))
mesh = ax.imshow(mask_edit, origin='lower', cmap=cmap, interpolation='none')

# Add gridlines
ax.set_xticks(np.arange(-0.5, mask_edit.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, mask_edit.shape[0], 1), minor=True)
ax.grid(which='minor', color='gray', linewidth=0.1)
ax.set_title("Click on cells to toggle land/ocean. Close figure when done.")

# Function to toggle cell on click
def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    j = int(event.xdata)
    i = int(event.ydata)
    if i<0 or i>=mask_edit.shape[0] or j<0 or j>=mask_edit.shape[1]:
        return
    # Toggle mask
    mask_edit[i, j] = 1 - mask_edit[i, j]
    mesh.set_data(mask_edit)
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Update ROMS grid after editing
hgrd.mask_rho = mask_edit
h[hgrd.mask_rho == 0] = hmin


pyroms.grid.write_ROMS_grid(grd, filename='INDO_1_12deg_grd.nc')
'''
print("Grid successfully written to INDO_1_12deg_grd.nc")
