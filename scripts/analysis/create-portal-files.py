#!python
"""Create files for portal ingestion from total hazard files."""
import datetime
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from geopandas.io.file import infer_schema

from vessel_drift_analysis.esi import ESI, clean_esi_string


logging.basicConfig(format='%(process)d - %(levelname)s: %(message)s', level=logging.INFO)


def esi_id_to_region(esi_id: str) -> str:
    """Given esi_id, return the region name."""
    regions = {
        'aleutians': 'Aleutians',
        'bristolbay': 'Bristol Bay',
        'cookinlet': 'Cook Inlet',
        'kodiak': 'Kodiak',
        'northslope': 'North Slope',
        'nwarctic': 'Northwest Arctic',
        'pwsound': 'Prince William Sound',
        'se': 'Southeast',
        'w': 'Western'
    }
    return regions[esi_id.split('-')[0]]


def process_monthly_file(monthly_file: Path, esi: ESI) -> gpd.GeoDataFrame:
    """Given path to monthly hazard file, return a GeoDataFrame prepped for portal ingestion.

    Parameters
    ----------
    monthly_file: Path
        Path to a file of a GeoDataFrame of spill results of all simulations for a month.


    Notes
    -----
    - Adds `spill_risk` defined as `spill_hazard` * ESI (0, 1]
    - Seperates risk for each vessel type and all vessel types combined
    """
    # total-hazard-month_<date>.parquet
    date = monthly_file.name.split('.')[0].split('_')[-1]
    date = datetime.datetime.strptime(date, '%Y-%m-%d')

    monthly_data = gpd.read_parquet(monthly_file)
    # Add risk estimate
    # Divide ESI by 10 to place in range 0 - 1.0
    monthly_data['spill_risk'] = monthly_data['hz_s'] * monthly_data['esi'] / 10.0

    # Weight results for month by the number of simulations
    nsims = len(monthly_data.date.unique())
    weighted_data = (monthly_data
        .groupby(['vessel_type', 'esi_id'])
        .agg({'breach_hazard': 'sum', 'hz_s': 'sum', 'spill_risk': 'sum'})
        .rename(columns={'hz_s': 'spill_hazard'})
    ) / nsims

    # Prepare subsets for each vessel type
    weighted_vessel_data = {}
    vessel_types = ['cargo', 'other', 'passenger', 'tanker']
    for vessel_type in vessel_types:
        df1 = (weighted_data
            .reset_index()
            .query(f'vessel_type=="{vessel_type}"')
            .set_index('esi_id'))

        df2 = esi.gdf.set_index('esi_id')
        df2.sort_index(inplace=True)

        tmp = (df2.join(
                df1.query(f'vessel_type=="{vessel_type}"')
            )
            .fillna(0)
            .drop(columns=['length', 'vessel_type'])
        )
        # Fill in missing values for "vessel_type" from the join
        tmp['vessel_type'] = vessel_type
        # I guess Oikos or the front end requires date to be formatted this way
        tmp['date_utc'] = date.strftime('%Y-%m-%dT00:00:00')
        tmp['region'] = [esi_id_to_region(esi_id) for esi_id in tmp.index]
        tmp = tmp.reset_index()
        columns = [
            'date_utc',
            'vessel_type',
            'region',
            'esi_id',
            'esi',
            'breach_hazard',
            'spill_hazard',
            'spill_risk',
            'geometry'
        ]
        tmp = tmp[columns]
        weighted_vessel_data[vessel_type] = tmp

    # Calculate results for 'all'
    all_tmp = weighted_vessel_data[vessel_types[0]].copy()
    for column in ['breach_hazard', 'spill_hazard', 'spill_risk']:
        for vessel_type in vessel_types[1::]:
            all_tmp[column] += weighted_vessel_data[vessel_type][column]
    all_tmp['vessel_type'] = 'all'
    weighted_vessel_data['all'] = all_tmp

    # Concat all values for a month into a single DataFrame
    return pd.concat(weighted_vessel_data.values())


def main(monthly_file_dir: Path, esi_path: Path, output_dir: Path, geojson: bool = False):
    """Create combined files of hazard and risk for use in portal."""
    # Load ESI, but use cleaned up values for ESI (max value as int)
    esi = ESI(esi_path)
    esi.gdf['esi'] = [clean_esi_string(esi_str) for esi_str in esi.gdf.esi]

    files = list(monthly_file_dir.glob('total-hazard-month_*.parquet'))
    files.sort()
    monthly_dfs = []
    for file in files:
        logging.info(f'Processing {file}')
        monthly_dfs.append(process_monthly_file(file, esi))

    combined = pd.concat(monthly_dfs)

    # Finishing touches: CRS, simplifying lines, schema, etc.
    # Need to be explicit
    combined.set_crs(epsg=4326)
    # Defines tolerance of simplification in native projection
    simplified_geometry = combined.simplify(tolerance=0.0001)
    combined.geometry = simplified_geometry
    logging.info(f'Size of combined {combined.memory_usage(deep=True).sum() / 1024 / 1024} MB')

    if geojson:
        out_path = output_dir / 'combined-hazard-risk-portal.geojson'
        logging.info(f'Saving combined data to {out_path}')
        combined.to_file(out_path, driver='GeoJSON')
    out_path = output_dir / 'combined-hazard-risk-portal_all.parquet'
    logging.info(f'Saving combined data to {out_path}')
    combined.to_parquet(out_path)

    # Now save files for each region
    regions = combined.region.unique()
    for region in regions:
        region_df = combined.query(f'region=="{region}"')

        if geojson or region == 'western':
            out_path = output_dir / f'combined-hazard-risk-portal_{region}.geojson'
            logging.info(f'Saving combined data for {region} to {out_path}')
            combined.to_file(out_path, driver='GeoJSON')
        region_name = region.lower().replace(' ', '-')
        out_path = output_dir / f'combined-hazard-risk-portal_{region_name}.parquet'
        logging.info(f'Saving combined data for {region} to {out_path}')
        region_df.to_parquet(out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'monthly_file_dir',
        type=Path,
        help='Path to directory with monthly hazard files'
    )
    parser.add_argument(
        'esi_path',
        type=Path,
        help='Path to ESI file'
    )
    parser.add_argument(
        'out_dir',
        type=Path,
        help='Path to directory to save outputs'
    )
    args = parser.parse_args()
    main(args.monthly_file_dir, args.esi_path, args.out_dir)