# Functions for plotting results from OpenDrift
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr

from .utils import get_stranded_flag_from_status


def plot_raster(path: Path, output_dir: Path, cmap: str = None, max_value: float = None) -> Path:
    """Make plot of raster (tif) and return path to image file"""
    with rasterio.open(path) as ds:
        dst_bounds = ds.bounds
        data = ds.read(1)

    # bounds of the AIS rasters
    # BoundingBox(left=-2965022.0, bottom=-180847.0, right=659978.0, top=2169153.0)
    # Adjust for plotting purposes using guess and check
    left = dst_bounds.left + 200_000
    top = dst_bounds.top + 300_000
    bottom = dst_bounds.bottom + 300_000
    right = dst_bounds.right

    if max_value is None:
        max_value = np.nanmax(data)

    # alpha to mask 0 values
    # 0: 0 whatever -> transparent
    # 1: != 1 -> opaque
    mask = data.copy()
    mask[mask != 0] = 1

    output_png_fname = path.name.split('.')[0] + '.png'
    output_png_fname = Path(output_dir) / output_png_fname

    aea_proj = ccrs.AlbersEqualArea(
        central_longitude=-154,
        central_latitude=50,
        standard_parallels=(55, 65)
    )

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=aea_proj)
    ax.coastlines()

    if cmap:
        base = plt.cm.get_cmap(cmap)
        color_list = base(np.linspace(0, 1, max_value))
        cmap_name = base.name + str(max_value)
        cmap = base.from_list(cmap_name, color_list, max_value)
    else:
        cmap = 'hot_r'

    p1 = ax.imshow(
        data,
        extent=(
            dst_bounds.left,
            dst_bounds.right,
            dst_bounds.bottom,
            dst_bounds.top
        ),
        transform=aea_proj,
        cmap=cmap,
        vmin=0,
        alpha=mask,
        vmax=max_value,
    )
    ax.add_feature(cartopy.feature.LAND.with_scale('110m'))
    ax.gridlines(draw_labels=True)
    ax.set_xlim([left, right])
    ax.set_ylim([bottom, top])
    fig.colorbar(p1, ax=ax, shrink=0.6)
    plt.savefig(output_png_fname)
    plt.close()


def plot_results(path: Path, output_dir: Path, skip: int = 100) -> Path:
    """Make scatter plot of results from OpenDrift and return path to image file."""
    with xr.open_dataset(path) as ds:
        lons = ds.lon.values
        lats = ds.lat.values
        status = ds.status.values
        stranded_flag = get_stranded_flag_from_status(ds)

    _ = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent(
        [-210, -140, 45, 75],
        crs=ccrs.PlateCarree()
    )
    nparticles = len(lons)
    if nparticles > skip:
        iter_particles = np.arange(0, nparticles, skip)
    else:
        iter_particles = np.arange(0, nparticles)

    for particle in iter_particles:
        # Create mask for stranded or not moving particles
        mask = np.logical_and(
            status[particle, :] <= 1,
            status[particle, :] >= 0
        )
        # Cannot save figure if you plot something that is completely masked
        # Cartopy will raise an error on save
        if not np.any(mask):
            continue
        ax.plot(
            lons[particle, mask],
            lats[particle, mask],
            color='g',
            transform=ccrs.PlateCarree()
        )

        # Starting position
        ax.scatter(
            ds.lon[particle, 0],
            ds.lat[particle, 0],
            c='b',
            transform=ccrs.PlateCarree()
        )
        stranded_ix = ds.status[particle, :] == stranded_flag

        # Stranded particles
        if np.any(stranded_ix):
            ax.scatter(
                ds.lon[particle, stranded_ix],
                ds.lat[particle, stranded_ix],
                c='r',
                transform=ccrs.PlateCarree()
            )

    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND.with_scale('110m'))
    ax.gridlines(draw_labels=True)

    fname = path.name.replace('.nc', '.png')
    outpath = output_dir / fname

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

    return outpath
