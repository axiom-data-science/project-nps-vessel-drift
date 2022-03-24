#!python
"""Create monthly hazard and risk files from individual simulation files."""
import datetime
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


def get_sim_dates(total_hazard_dir: Path) -> list:
    """Given path to individual hazard results, return sims dates from file names."""
    dates = []
    for file in total_hazard_dir.glob('total-hazard_*.parquet'):
        # total-hazard_<%Y-%m-%d>.parquet
        date = file.name.split('.')[0].split('_')[-1]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        dates.append(date)

    dates = list(set(dates))
    dates.sort()

    return dates


def load_month_of_hazard_results(sim_dates: list, month_num: int, hazard_files: list) -> gpd.GeoDataFrame:
    """Given list of sim dates, hazard files, and a month number, return a DataFrame with results"""
    dates = [date for date in sim_dates if date.month == month_num]

    gdfs = []
    for hazard_file in hazard_files:
        file_date = hazard_file.name.split('.')[0].split('_')[-1]
        file_date = datetime.datetime.strptime(file_date, '%Y-%m-%d')
        if file_date in dates:
            gdfs.append(gpd.read_parquet(hazard_file))

    return pd.concat(gdfs)


def load_and_save_monthly_results(hazard_results_dir: Path, outdir: Path, geojson=False) -> None:
    """Given path to individual sime results dir load and save data into monthly files."""
    # Get all hazard files
    total_hazard_files = list(hazard_results_dir.glob('total-hazard_2019*parquet'))
    total_hazard_files.sort()

    # Get simulation dates from file names
    sim_dates = get_sim_dates(hazard_results_dir)

    for month_num in range(1, 13):
        logging.info(f'Loading results from month: {month_num}')
        monthly_data = load_month_of_hazard_results(sim_dates, month_num, total_hazard_files)
        outpath = outdir / f'total-hazard-month_2019-{month_num:02d}-01.parquet'
        logging.info(f'Saving results to {outpath}')
        monthly_data.to_parquet(outpath)
        if geojson:
            outpath = outdir / f'total-hazard-month_2019-{month_num:02d}-01.geojson'
            logging.info(f'Saving results to {outpath}')
            monthly_data.to_file(outpath, driver='GeoJSON')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'hazard_dir',
        type=Path,
        help='Path to dir with individual simulation hazard files'
    )
    parser.add_argument(
        'out_dir',
        type=Path,
        help='Path to output dir'
    )
    args = parser.parse_args()
    hazard_dir = args.hazard_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=False)

    logging.info(f'Binning individual simulation hazard files from {hazard_dir}')
    logging.info(f'Saving binned results to {hazard_dir}')

    load_and_save_monthly_results(hazard_dir, out_dir)
