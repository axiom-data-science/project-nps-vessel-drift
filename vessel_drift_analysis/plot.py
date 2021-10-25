# Functions for plotting results from OpenDrift
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .utils import get_stranded_flag_from_status


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
