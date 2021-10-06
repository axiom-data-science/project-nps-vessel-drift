#!python
"""Script to process AIS files already processed using Spark based methods for drift simulations.

Inputs are a directory of AIS files processed into heatmaps from the Spark-based method.
Outputs are a directory of AIS heatmaps standardized to a common grid and reprojected to EPSG:4326.
"""
import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path

import affine

import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


@contextmanager
def resample_raster(raster, out_path=None, scale=2):
    """ Resample a raster
        multiply the pixel size by the scale factor
        divide the dimensions by the scale factor
        i.e
        given a pixel size of 250m, dimensions of (1024, 1024) and a scale of 2,
        the resampled raster would have an output pixel size of 500m and dimensions of (512, 512)
        given a pixel size of 250m, dimensions of (1024, 1024) and a scale of 0.5,
        the resampled raster would have an output pixel size of 125m and dimensions of (2048, 2048)
        returns a DatasetReader instance from either a filesystem raster or MemoryFile (if out_path is None)
    """
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a * scale, t.b, t.c, t.d, t.e * scale, t.f)
    height = int(raster.height / scale)
    width = int(raster.width / scale)

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read(
            out_shape=(raster.count, height, width),
            resampling=Resampling.bilinear,
        )

    if out_path is None:
        with write_mem_raster(data, **profile) as dataset:
            del data
            yield dataset

    else:
        with write_raster(out_path, data, **profile) as dataset:
            del data
            yield dataset


@contextmanager
def write_mem_raster(data, **profile):
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return


@contextmanager
def write_raster(path, data, **profile):

    with rasterio.open(path, 'w', **profile) as dataset:  # Open as DatasetWriter
        dataset.write(data)

    with rasterio.open(path) as dataset:  # Reopen as DatasetReader
        yield dataset


def _make_rescale_name(input_raster: Path, version: int = 2) -> str:
    # Terrestial AIS data
    if version == 1:
        fname = input_raster.name
        words = fname.split('.')[1].split('_')
        # shiptype_month_dataset.png
        output_fname = '_'.join(words[8:11]) + '.tif'
    # Satellite AIS data
    elif version == 2:
        fname = input_raster.name
        ship_type, date, sum_type, _ = fname.split('.')[0].split('_')
        output_fname = '_'.join([ship_type, date, sum_type]) + '.tif'
    else:
        raise ValueError('version must be 1 or 2')

    return output_fname


def _find_overlapping_bounds(dir: Path) -> tuple[float, float, float, float]:
    """
    Find the common bounds from a directory of rasters.

    Parameters
    -----------
    dir: Path
        Path to directory of AIS rasters.

    Returns
    --------
    bounds: tuple
        Bounds of all tifs (xmin, ymin, xmax, ymax). The largest xmin/ymin and smallest xmax/ymax.

    Notes
    ------
    raster bounds are in the format:
    (left, bottom, right, top)
    (xmin, ymin, xmax, ymax)
    """
    smallest_xmin = 9999999
    smallest_ymin = 9999999
    largest_xmax = -9999999
    largest_ymax = -9999999

    paths = Path(dir).glob('*.tif')
    for path in paths:
        with rasterio.open(path) as ds:
            xmin, ymin, xmax, ymax = ds.bounds
        if xmin < smallest_xmin:
            smallest_xmin = xmin
        if ymin < smallest_ymin:
            smallest_ymin = ymin
        if xmax > largest_xmax:
            largest_xmax = xmax
        if ymax > largest_ymax:
            largest_ymax = ymax

    bounds = (smallest_xmin, smallest_ymin, largest_xmax, largest_ymax)
    print(f'{bounds=}')

    return bounds


def get_vrt_options(input_dir: Path) -> Path:
    """Return vrt options from input AIS files used for standardizing raster.

    Parameters
    -----------
    input_dir: Path
        Path to directory of AIS rasters.
    cell_size: int
        Cell size of output raster.

    Returns
    --------
    vrt_options: dict
        VRT options for reprojecting input rasters to a standard grid.
    """
    # Need a sample file to get resolution of input rasters
    sample_path = Path(input_dir).glob('*.tif')[0]
    with rasterio.open(sample_path) as ds:
        dst_crs = ds.crs
        if ds.res[0] != ds.res[1]:
            raise ValueError('Input rasters are not square.  Not designed to handle these rasters.')
        cell_size = ds.res[0]

    # Get bounds from all rasters -> find largest xmin/ymin and smallest xmax/ymax
    # Used to get overlapping region of all rasters
    xmin, ymin, xmax, ymax = _find_overlapping_bounds(input_dir)

    # Set affine transformation for the desired standardized raster
    left = xmin
    top = ymax
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    dst_width = xdelta / cell_size
    dst_height = ydelta / cell_size
    dst_transform = affine.Affine(
        cell_size, 0.0, left,
        0.0, -cell_size, top
    )

    vrt_options = {
        'resampling': Resampling.bilinear,
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width
    }

    return vrt_options


def standardize(input_fname: Path, output_dir: Path, vrt_options: dict, ais_version: int = 2) -> Path:
    """Given a tif, rescale it to the standardized bounds.

    Parameters
    -----------
    input_fname: Path
        Path to input raster.
    output_dir: Path
        Path to output directory to save standardized raster.
    vrt_options: dict
        VRT options for reprojecting input rasters to a standard grid.
    ais_version: int
        Version of AIS data.  1 for Terrestial AIS data, 2 for Satellite AIS data. (defines path)

    Returns
    --------
    output_fname: Path
        Path to standardized raster.
    """
    output_fname = _make_rescale_name(input_fname, ais_version)
    output_fname = Path(output_dir) / output_fname

    with rasterio.open(input_fname) as src:
        with WarpedVRT(src, **vrt_options) as vrt:
            data = vrt.read()  # noqa 

            for _, window in vrt.block_windows():
                data = vrt.read(window=window)  # noqa

            print(f'saving to {output_fname}')
            rio_shutil.copy(vrt, output_fname, driver='GTiff')

    return output_fname


def rescale(input_fname: Path, output_dir: Path, scale: float = 50) -> Path:
    """Given a tif, rescale it to desired resolution.

    Parameters
    -----------
    input_fname: Path
        Path to input raster.
    output_dir: Path
        Path to output directory to save standardized raster.
    scale: float
        Scale factor for output raster.

    Returns
    --------
    output_fname: Path
        Path to standardized raster.

    Examples
    ---------
    >>> rescale(input_fname, output_dir, scale=50) # scale input res by 50 (e.g. 100m -> 5000m)
    output_dir / input_fname.name
    """
    output_tif_fname = Path(output_dir) / input_fname.name

    with rasterio.Env():
        with rasterio.open(input_fname) as ds1:
            with resample_raster(ds1, scale=scale) as ds:
                data = ds.read(1)
                data = np.ma.masked_where(data == -2147483648, data)

                profile = ds.profile
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    compress='lzw',
                    nodata=0
                )

                data = data.data
                data[data == -2147483648] = 0
                print(f'saving to {output_tif_fname}')
                with rasterio.open(output_tif_fname, 'w', **profile) as dst:
                    dst.write(data.astype(rasterio.uint8), 1)

    return output_tif_fname


def reproject_to_epsg4326(input_fname, output_dir, method='gdal'):
    """Reproject input raster to EPSG:4326.

    Parameters
    -----------
    input_fname: Path
        Path to input raster.
    output_dir: Path
        Path to output directory to save raster with EPSG:4326 projection.
    method: str
        Method to reproject raster.  'gdal' or 'rasterio'.

    Returns
    --------
    output_fname: Path
        Path to raster with EPSG:4326 projection.

    Notes
    ------
    Rasterio method appears to introduce artifacts in the output.
    """
    if method == 'gdal':
        output_fname = _reproject_to_epsg4326_gdal(input_fname, output_dir)
    elif method == 'rasterio':
        output_fname = _reproject_to_epsg4326_rasterio(input_fname, output_dir)
    else:
        raise ValueError('method must be gdal or rasterio')

    return output_fname


def _reproject_to_epsg4326_gdal(
    input_fname: Path,
    output_dir: Path,
    method: str = 'bilinear',
    res: float = 0.225
) -> Path:
    """Reproject input raster to EPSG:4326 using gdal.

    Parameters
    -----------
    input_fname: Path
        Path to input raster.
    output_dir: Path
        Path to output directory to save raster with EPSG:4326 projection.
    method: str
        Method to reproject raster.  'bilinear' or 'cubic'.
    res: float
        Resolution of output raster.

    Returns
    --------
    output_fname: Path
        Path to raster with EPSG:4326 projection.
    """
    output_tif_fname = Path(output_dir) / input_fname.name
    cmd = f'gdalwarp -t_srs EPSG:4326 -r bilinear -tr 0.225 0.25 {input_fname} {output_tif_fname}'
    cmd = [c.strip() for c in cmd.split(' ')]
    subprocess.run(cmd, capture_output=True)

    return output_tif_fname


def _reproject_to_epsg4326_rasterio(input_fname: Path, output_dir: Path) -> Path:
    """Reproject input raster to EPSG:4326 using rasterio.

    Parameters
    -----------
    input_fname: Path
        Path to input raster.
    output_dir: Path
        Path to output directory to save raster with EPSG:4326 projection.

    Returns
    --------
    output_fname: Path
        Path to raster with EPSG:4326 projection.

    Notes
    ------
    Rasterio method appears to introduce artifacts in the output.
    """
    output_tif_fname = Path(output_dir) / input_fname.name

    with rasterio.open(input_fname) as src:
        transform, width, height = calculate_default_transform(
            src.crs, 'epsg:4326', src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'epsg:4326',
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif_fname, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='epsg:4326',
                    resampling=Resampling.bilinear
                )

    return output_tif_fname


def plot(input_fname: Path, output_dir: Path, max_value: int = 5) -> None:
    """Plot raster.

    Parameters
    ----------
    input_fname: Path
        Path to input raster.
    output_dir: Path
        Path to output directory to save image.
    max_value: int
        Maximum value to plot on colorbar.
    """
    with rasterio.open(input_fname) as ds:
        dst_bounds = ds.bounds
        data = ds.read(1)

    output_png_fname = input_fname.name.split('.')[0] + '.png'
    output_png_fname = Path(output_dir) / output_png_fname

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
        vmax=max_value,
    )
    ax.add_feature(cartopy.feature.LAND.with_scale('110m'))
    ax.gridlines(draw_labels=True)
    fig.colorbar(p1, ax=ax, shrink=0.6)
    plt.savefig(output_png_fname)
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        type=str,
        help='path to file or directory'
    )
    parser.add_argument(
        'output',
        type=str,
        help='path to output directory'
    )
    parser.add_argument(
            '--rescale',
            type=int,
            default=10,
            help='divisor for rescaling original resolution (e.g. 500 m * rescale)'
    )
    parser.add_argument(
            '--ais_version',
            type=int,
            default=2,
            help='ais type (sat [1] or terrestrial [2])'
    )
    parser.add_argument(
            '--number',
            type=int,
            default=None,
            help='Limit number of tifs processed'
    )

    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if input_path.is_dir():
        files = input_path.glob('*.tif')
    else:
        files = [input_path]

    if args.number:
        files = list(files)[:args.number]

    ais_type = 'satellite' if args.ais_version == 1 else 'terrestrial'

    logger = logging.getLogger(__name__)
    logger.info(f'Processing AIS files from {input_path} of type {ais_type}')
    logger.info('Parameters:')
    logger.info(f'{input_path=}')
    logger.info(f'{output_path=}')
    logger.info(f'{rescale=} (rescaling factor)')
    if args.number:
        logger.info(f'Processing {args.number} files. ({args.number=}') 

    nfiles = len(files)
    for i, file in enumerate(files):
        if file.is_symlink():
            continue

        logger.info(f'Processing {i} of {nfiles}: {file}')
        if i == 0:
            vrt_options = get_vrt_options(file)

        outdir = output_path / 'raw-images'
        outdir.mkdir(exist_ok=True, parents=True)
        plot(file, outdir)

        outdir = output_path / 'standard'
        outdir.mkdir(exist_ok=True, parents=True)
        standardize_fname = standardize(file, outdir, vrt_options, ais_version=args.ais_version)

        outdir = output_path / 'standardize-images'
        outdir.mkdir(exist_ok=True, parents=True)
        plot(standardize_fname, outdir)

        outdir = output_path / 'rescale'
        outdir.mkdir(exist_ok=True, parents=True)
        rescale_fname = rescale(standardize_fname, outdir, args.rescale)

        outdir = output_path / 'rescale-images'
        outdir.mkdir(exist_ok=True, parents=True)
        plot(rescale_fname, outdir)

        outdir = output_path / 'epsg4326'
        outdir.mkdir(exist_ok=True, parents=True)
        epsg4326_fname = reproject_to_epsg4326(rescale_fname, outdir)

        outdir = output_path / 'epsg4326-images'
        outdir.mkdir(exist_ok=True, parents=True)
        plot(epsg4326_fname, outdir)
