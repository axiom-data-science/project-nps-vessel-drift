#!python
"""Reproject NAM files for NPS vessel drift to lan/lon for use with OpenDrift"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyproj
from pyproj import Transformer
import xarray as xr


NAM_PROJ = pyproj.crs.CRS.from_proj4('+proj=stere +lat_0=90 +lon_0=210 +k_0=0.9330127018922193 +R=6371229.0')


def get_files(dir: str, glob_str: str) -> List[Path]:
    files = Path(dir).glob(glob_str)
    files = list(files)
    files.sort()

    return files


def _nam_meshgrid(fname: str) -> Tuple[np.array, np.array]:
    """Given NAM file path, return meshgrid of locations in m"""
    with xr.open_dataset(fname) as ds:
        # multiple by 1000 to convert from km to m
        xx, yy = np.meshgrid(ds.x * 1000, ds.y * 1000)

    return xx, yy


def _nam_latlons(
    fname: str,
    src_crs: pyproj.crs.CRS = NAM_PROJ,
    dst_crs: str = "EPSG:4326"
) -> Tuple[np.array, np.array]:
    """Given NAM file path, return meshgrid of locations in lat/lon [0, 360]"""
    xx, yy = _nam_meshgrid(fname)
    xformer = Transformer.from_crs(src_crs, dst_crs)
    lat, lon = xformer.transform(xx, yy)
    lon = lon % 360

    return lat, lon


def _fixtimes(ds: xr.Dataset) -> Tuple[np.array, np.array, np.array]:
    """Given a NAM dataset, fix the arrays and return as type"""
    ixs = np.argsort(ds.time.values)
    new_time = np.zeros_like(ds.time)
    new_u = np.zeros_like(ds.wind_u)
    new_v = np.zeros_like(ds.wind_v)
    for i, ix in enumerate(ixs):
        new_time[i] = ds.time.values[ix]
        new_u[i, :, :] = ds.wind_u[ix, :, :]
        new_v[i, :, :] = ds.wind_v[ix, :, :]

    return new_time, new_u, new_v


def _reproject_file(
    fname: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    outfile_prefix: str = 'reproj_'
) -> None:
    outdir = fname.parent
    outname = outfile_prefix + fname.name
    out_fname = outdir / outname
    with xr.open_dataset(fname) as src_ds:
        new_time, new_u, new_v = _fixtimes(src_ds)
        ds = xr.Dataset(
            {
                "wind_u": (
                    ("time", "y", "x"),
                    # 10 m air
                    new_u,
                    {
                        "_FillValue": np.nan,
                        "missing_value": np.nan,
                        "units": "m.s-1",
                        "standard_name": "eastward_wind",
                        "long_name": "Eastward wind"
                    }
                ),
                "wind_v": (
                    ("time", "y", "x"),
                    # 10 m air
                    new_v,
                    {
                        "_FillValue": np.nan,
                        "missing_value": np.nan,
                        "units": "m.s-1",
                        "standard_name": "northward_wind",
                        "long_name": "Northward wind"
                    }
                ),
                "lat": (
                    ("y", "x"),
                    lat,
                    {
                        "standard_name": "latitude",
                        "units": "degree_north",
                        "long_name": "latitude"
                    }
                ),
                "lon": (
                    ("y", "x"),
                    lon,
                    {
                        "standard_name": "longitude",
                        "units": "degree_east",
                        "long_name": "longitude"
                    }
                ),
                "time": (
                    ("time"),
                    # convert from ns to seconds
                    new_time.astype(int)/10**9,
                    {
                        "standard_name": "time",
                        "units": "seconds since 1970-01-01 00:00:00",
                        "long_name": "time"
                    }
                )
            },
        )
        ds.to_netcdf(out_fname)


def reproject_files(dir):
    files = get_files(dir, 'alaska_hires_2019-*.nc')
    lat, lon = _nam_latlons(files[0])
    for file in files:
        print(file)
        _reproject_file(file, lat, lon)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_dir',
        type=str,
        help='directory with NAM data'
    )
    args = parser.parse_args()
    print(f'Reprojecting data in {args.data_dir}')
    reproject_files(args.data_dir)
