#!python
"""Used to create rasters from outputs of OpenDrift simulation results."""
import logging
from pathlib import Path

import geopandas
import numpy as np
import rasterio
import xarray as xr
from rasterio import features

# All analysis performed on the 25 km x 25 km grid in Alaska Albers Equal Area Projection
REFERENCE_TIF = '/mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2015-2020/processed_25km/2019/rescale/all_20190101-20190201_total.tif'



def get_stranded_flag(ds: xr.Dataset) -> int:
    """Given a Dataset, return the integer flag indicating 'stranded' status."""
    flag_meanings = ds.status.flag_meanings.split(' ')
    for ix, flag_meaning in enumerate(flag_meanings):
        if flag_meaning == 'stranded':
            return ix


def get_stranded_locs(ds: xr.Dataset) -> int:
    """Given a Dataset, return the stranded locations (lon, lat) as ndarray (npoints, 2)."""
    stranded_flag = get_stranded_flag(ds)
    stranded_ix = ds.status == stranded_flag
    lons = ds.lon.values[stranded_ix]
    lats = ds.lat.values[stranded_ix]
    
    return np.vstack([lons, lats]).T


def lon_to_epsg4326(lon: np.ndarray) -> np.ndarray:
    """Given lon in (0, 360) reference, return lon in (-180, 180)"""
    return np.mod(lon - 180, 360) - 180


def load_sim_output(fpath: Path, out_crs: str) -> geopandas.GeoDataFrame:
    """Given path to OpenDrift output and WKT string, return results as GeoDataFrame"""
    with xr.open_dataset(fpath) as ds:
        df = ds.to_dataframe()

    # lon in (-180, 180)
    df.lon = lon_to_epsg4326(df.lon)
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=geopandas.points_from_xy(df.lon, df.lat)
    )
    # need to explicitly set CRS so it can be correclty converted to Alaska AEA
    gdf = gdf.set_crs('epsg:4326')

    return gdf.to_crs(out_crs)


def rasterize_sim_result(fpath: Path, out_fpath: Path, ref_tif_fpath: Path) -> None:
    """Given path to simulation result, rasterize and save as GeoTiff"""
    with rasterio.open(ref_tif_fpath) as ref_tif:
        meta = ref_tif.meta.copy()

        gdf = load_sim_output(fpath, ref_tif.crs.to_wkt())
        raster = features.rasterize(
            gdf.geometry,
            out_shape=ref_tif.shape,
            all_touched=True,
            transform=ref_tif.transform,
            merge_alg=rasterio.enums.MergeAlg.add
        )

        with rasterio.open(out_fpath, 'w+', **meta) as out_ds:
            # out array needs to be of shape: (nbands, height, width)
            # - raster is (height, width)
            out_ds.write(raster[np.newaxis, :, :])


def rasterize_sim_strandings(fpath: Path, out_fpath: Path, ref_tif_fpath: Path) -> None:
    """Given path to simulation result, rasterize and save strandings as GeoTiff"""
    with rasterio.open(ref_tif_fpath) as ref_tif:
        meta = ref_tif.meta.copy()

        gdf = load_sim_output(fpath, ref_tif.crs.to_wkt())

        # only keep the stranded positions for the raster
        with xr.open_dataset(fpath) as ds:
            stranding_flag = get_stranded_flag(ds)
        stranded_mask = gdf.status == stranding_flag
        gdf = gdf[stranded_mask]

        raster = features.rasterize(
            gdf.geometry,
            out_shape=ref_tif.shape,
            all_touched=True,
            transform=ref_tif.transform,
            merge_alg=rasterio.enums.MergeAlg.add
        )

        with rasterio.open(out_fpath, 'w+', **meta) as out_ds:
            # out array needs to be of shape: (nbands, height, width)
            # - raster is (height, width)
            out_ds.write(raster[np.newaxis, :, :])


def main(results_dir: Path, output_dir: Path, stranding: bool=False, ref_tif_fpath: Path=REFERENCE_TIF) -> None:
    """Given paths to sim results and output dir, create rasters of output drift simulations"""
    sim_files = list(results_dir.glob('*nc'))
    sim_files.sort()

    if not output_dir.exists():
        output_dir.mkdir()

    for sim_file in sim_files:
        out_fname = sim_file.name.replace(sim_file.suffix, '.tif')
        if stranding:
            out_fname = 'stranding_' + out_fname
        out_fpath = output_dir / out_fname

        logging.info(f'Creating raster of {sim_file} saved to {out_fpath}')
        if stranding:
            rasterize_sim_strandings(sim_file, out_fpath, ref_tif_fpath)
        else:
            rasterize_sim_result(sim_file, out_fpath, ref_tif_fpath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to directory with model results'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to directory to save results'
    )
    parser.add_argument(
        '-s',
        '--stranding',
        action='store_true',
        help='Only rasterize the stranding locations'
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir) 
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True) 
    main(results_dir, output_dir, args.stranding)
