#!python
# Plot rasters
import logging
from pathlib import Path

from vessel_drift_analysis import plot


def main(input_dir: Path, output_dir: Path, file_suffix: str = 'tif', cmap: str = None, max_value: float = None) -> None:
    """Plot results from drift simulations.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the simulation results.
    output_dir : str
        Path to the directory where the plots will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Plotting results from {input_dir} and saving to {output_dir}')

    input_files = input_dir.glob(f'./*{file_suffix}')
    for input_file in input_files:
        plot.plot_raster(input_file, output_dir, cmap, max_value)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot a directory of tifs (or any raster that rasterio can read).'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Path to the directory containing the rasters.'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Path to the directory where the plots will be saved.'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='tif',
        help='File suffix if not loading "tif" files (e.g. "tiff", etc.).'
    )
    parser.add_argument(
        '--max',
        type=float,
        default=None,
        help='Max value for colorbar'
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default=None,
        help='Matplotlib colormap'
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.suffix, args.max, args.cmap)
