#!python
"""Calculate total hazard and risk and persist terms in GeoJSON + Parquet files."""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from vessel_drift_analysis.esi import clean_esi_string

logging.basicConfig(format='%(process)d - %(levelname)s: %(message)s', level=logging.INFO)


def combine_hazard_results(drift_hazard: Path, spill_hazard: Path, esi: Path) -> gpd.GeoDataFrame:
    """Given drift hazard file, spill hazard file, and ESI file, return a combined GeoDataFrame."""
    drift_df = pd.read_parquet(drift_hazard)
    # Need breach hazard, so we add that to the dataframe
    drift_df['breach_hazard'] = drift_df.stranding_hazard * drift_df.breach_prob
    # Group by date, vessel_type, and ESI segment to combine with spill results
    breach_hazard = (
        drift_df
        .groupby(['date', 'vessel_type', 'esi_id'])
        .agg({'breach_hazard': 'sum'})
    )
    # Don't need starts for non-stranding vessels, so we filter those out
    breach_hazard = breach_hazard[breach_hazard.breach_hazard > 0]

    # Group the same way
    spill_df = pd.read_parquet(spill_hazard)
    spill_hazard = (
        spill_df
        .groupby(['date', 'vessel_type', 'esi_id'])
        .agg({'oil_mass': 'sum', 'cs': 'sum', 'pb': 'sum'})
    )

    total_hazard = breach_hazard.join(spill_hazard)
    # Actually add the "total hazard" Hz_s
    # Probability of a breach * # contacts in ESI segment / total particles * concentration index
    total_hazard['hz_s'] = total_hazard.breach_hazard * total_hazard.pb * total_hazard.cs

    # Add ESI information
    esi = gpd.read_parquet(esi)
    total_hazard_with_esi = pd.merge(total_hazard.reset_index(), esi, on='esi_id')
    # "clean" esi values -> take the maximum for worst case scenarios
    total_hazard_with_esi['esi'] = [clean_esi_string(esi) for esi in total_hazard_with_esi.esi]
    # need to convert datetime.date to datetime
    total_hazard_with_esi['date'] = pd.to_datetime(total_hazard_with_esi.date)
    # Many ESI segments that were not hit are filled with NaNs, change that to 0
    total_hazard_with_esi.fillna(0, inplace=True)
    total_hazard_with_esi = gpd.GeoDataFrame(total_hazard_with_esi, geometry='geometry')

    return total_hazard_with_esi


def main(drift_hazard: Path, spill_hazard: Path, esi: Path, out_dir: Path) -> None:
    """Write out combined total hazard factors to GeoJSON and Parquet files"""
    total_hazard = combine_hazard_results(drift_hazard, spill_hazard, esi)

    # Can't write out the thing as a single file because it exhausts memory
    # So, we write out the data by simulation date for all vessel types
    for date in total_hazard.date.unique():
        date = np.datetime_as_string(date, 'D')
        outpath = out_dir / f'total-hazard_{date}.parquet'
        logging.info(f'Writing out {outpath}')
        (total_hazard
            .query(f'date=="{date}"')
            .to_parquet(outpath))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'drift_hazard',
        type=Path,
        help='Drift hazard Parquet file'
    )
    parser.add_argument(
        'spill_hazard',
        type=Path,
        help='Spill hazard Parquet file'
    )
    parser.add_argument(
        'esi',
        type=Path,
        help='ESI parquet file'
    )
    parser.add_argument(
        'out_dir',
        type=Path,
        help='Output directory'
    )
    args = parser.parse_args()

    logging.info(f'Reading drift hazard from {args.drift_hazard.resolve()}')
    logging.info(f'Reading spill hazard from {args.spill_hazard.resolve()}')
    logging.info(f'Reading ESI from {args.esi.resolve()}')
    logging.info(f'Writing output to {args.out_dir.resolve()}')

    main(args.drift_hazard, args.spill_hazard, args.esi, args.out_dir)
