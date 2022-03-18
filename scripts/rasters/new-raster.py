#!python
"""Create rasters of outputs of OpenDrift simulation results."""
import logging
from pathlib import Path

import cartopy
import pyproj
import rasterio
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# From /mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2015-2020/processed_25km/2019/rescale/all_20190101-20190201_total.tif
# 25 km x 25 km AIS data in Alaska Albers equal area proj
TIF_META = {
    "bounds": [-2512022.0, 71153.0, 637978.0, 2871153.0],
    "colorinterp": ["gray"],
    "compress": "lzw",
    "count": 1, "crs":
    "EPSG:3338",
    "descriptions": [None],
    "driver": "GTiff",
    "dtype": "uint8",
    "height": 112,
    "indexes": [1],
    "interleave": "band",
    "lnglat": [-172.20128584797783, 62.053338076063866],
    "mask_flags": [["nodata"]],
    "nodata": 0.0,
    "res": [25000.0, 25000.0],
    "shape": [112, 126],
    "tiled": False,
    "transform": [25000.0, 0.0, -2512022.0, 0.0, -25000.0, 2871153.0, 0.0, 0.0, 1.0],
    "units": [None],
    "width": 126
}


def lon_to_epsg4326(lon: np.ndarray) -> np.ndarray:
    """Given lon in (0, 360) reference, return lon in (-180, 180)"""
    return np.mod(lon - 180, 360) - 180


def bin_results(lon, lat, bins, in_crs='epsg:4326', out_crs='epsg:3338'):
    """Given bins [x, y], lon, lat, return positions binned for raster."""
    in_proj = pyproj.Proj(init=in_crs)
    out_proj = pyproj.Proj(init=out_crs)
    x, y = pyproj.transform(
        in_proj,
        out_proj,
        lon,
        lat,
    )
    h, _, _ = np.histogram2d(x, y, bins=(bins.x, bins.y))

    # gotta rotate 90 to align with rasters correctly
    return np.rot90(h)


def plot_tif(binned, output_file, tif_meta=TIF_META):
    """Given data binned to tif raster shape, save figure at output_file"""
    dst_bounds = tif_meta['bounds']
    max_value = int(binned.max())

    aea_proj = ccrs.AlbersEqualArea(
        central_longitude=-154,
        central_latitude=50,
        standard_parallels=(55, 65)
    )

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=aea_proj)
    ax.coastlines()

    base = plt.cm.get_cmap('pink')
    color_list = base(np.linspace(0, 1, max_value))
    cmap_name = base.name + str(max_value)
    cmap = base.from_list(cmap_name, color_list, max_value)

    p1 = ax.imshow(
        binned,
        extent=(
            dst_bounds[0],
            dst_bounds[2],
            dst_bounds[1],
            dst_bounds[3]
        ),
        transform=aea_proj,
        cmap=cmap,
        vmin=0,
        vmax=max_value,
    )
    ax.add_feature(cartopy.feature.LAND.with_scale('110m'))
    ax.gridlines(draw_labels=True)
    fig.colorbar(p1, ax=ax, shrink=0.6)

    plt.savefig(output_file)
    plt.close()
