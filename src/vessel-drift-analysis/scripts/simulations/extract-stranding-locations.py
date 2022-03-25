#!python
# Extract stranding locations (lon, lat) from the results of OpenDrift simulations
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from vessel_drift_analysis.utils import get_stranded_flag_from_status


def get_stranded_locations(ds: xr.Dataset) -> np.ndarray:
    """Return locations (lon, lat) of stranded vessels."""
    stranded_flag = get_stranded_flag_from_status(ds)
    stranded_ix = ds.status == stranded_flag
    lons = ds.lon.values[stranded_ix]
    lats = ds.lat.values[stranded_ix]

    return np.vstack([lons, lats]).T


def save_stranded_locations(opendrift_file: Path, outdir: Path) -> Path:
    """Given path to OpenDrift result file, save stranding locations as csv to outdir."""
    outdir.mkdir(exist_ok=True, parents=True)
    outfile = outdir / opendrift_file.name.replace('.nc', '')

    with xr.open_dataset(opendrift_file) as ds:
        locs = get_stranded_locations(ds)

    np.save(outfile, locs)

    return outfile


def main(indir, outdir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f'Extracting stranding locations from {indir} and saving to {outdir}')

    results_files = indir.glob('*.nc')
    for result_file in results_files:
        try:
            logging.info(f'Extracting stranding locations from {result_file}')
            out_file = save_stranded_locations(result_file, outdir)
            logging.info(f'Extracted stranding locations saved to {out_file}')
        except:
            logging.error(f'Problem extracting location from {result_file}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'indir',
        type=Path,
        help='Directory with OpenDrift results'
    )
    parser.add_argument(
        'outdir',
        type=Path,
        help='Directory to save extracted stranding locations'
    )
    args = parser.parse_args()
    main(args.indir.resolve(), args.outdir.resolve())
