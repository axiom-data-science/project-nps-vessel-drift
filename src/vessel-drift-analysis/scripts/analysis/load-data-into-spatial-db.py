#!python
# Loads vessel hazard results into the spatial db
import logging
import os
from pathlib import Path

import geopandas as gpd
import sqlalchemy
from sqlalchemy import create_engine


logging.basicConfig(format='%(process)d - %(levelname)s: %(message)s', level=logging.INFO)


def get_db_engine() -> sqlalchemy.engine.Engine:
    spatial_db_password = os.environ['SPATIAL_DB_PASSWORD']
    db_url = f'postgresql://db1.axiomptk:5432/spatial?user=spatial&password={spatial_db_password}'
    return create_engine(db_url)


def load_file(hazard_file_path: Path, db_engine: sqlalchemy.engine.Engine) -> None:
    """Given a path to a hazard result file, load it, preprocess, and load into spatial db"""
    SCHEMA_NAME = 'axiom_nps_vessel_drift'

    def get_region(hazard_file_path: Path):
        # combined-hazard-risk-portal_<region>.parquet
        try:
            region = hazard_file_path.name.split('.')[0].split('_')[1]
        except IndexError:
            region = 'all'
        return region

    region = get_region(hazard_file_path)
    if region is None:
        region == 'all'
    layer_name = f'vessel_hazard_{region}'

    # Need to specify as DateTime otherwise it will be ingested as text
    dtypes = {
        'date_utc': sqlalchemy.DateTime
    }
    gdf = gpd.read_parquet(hazard_file_path)
    logging.info(f'Loading {hazard_file_path} as {layer_name}')
    gdf.to_postgis(
        layer_name,
        db_engine,
        schema=SCHEMA_NAME,
        if_exists='replace',
        chunksize=1000,
        dtype=dtypes
    )


def main():
    engine = get_db_engine()
    files_dir = Path('/mnt/store/data/assets/nps-vessel-spills/sim-results/satellite-sims/portal-files')
    logging.info(f'Loading files from {files_dir.resolve()}')
    files = list(files_dir.glob('combined-hazard-risk-portal*.parquet'))
    files.sort()
    for f in files:
        load_file(f, engine)


if __name__ == '__main__':
    main()
