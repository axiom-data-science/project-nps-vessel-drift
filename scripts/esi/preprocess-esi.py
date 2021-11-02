#!python
"""Convert shp files to geojson, combined, and save as parquet."""
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd


logging.basicConfig(format='%(asctime)-15s - %(levelname)s: %(message)s', level=logging.INFO)

RENAME_COLUMNS = {
    'ESILGEN_': 'esi_id',
    'ESI': 'esi',
    'LENGTH': 'length'
}
DROP_COLUMNS = [
    'FNODE_',
    'TNODE_',
    'LPOLY_',
    'RPOLY_',
    'SOURCE_ID',
    'ENVIR',
    'MOSTSENSIT',
    'LINE',
    'ESILGEN_ID',
]

base_dir = Path('/mnt/store/data/assets/nps-vessel-spills/spatial-division/esi/esi-shapefiles')
esil_files = base_dir.glob('*/**/esil.shp')
out_dir = Path('/mnt/store/data/assets/nps-vessel-spills/spatial-division/esi/cleaned-and-combined')
out_dir.mkdir(exist_ok=True)


def get_region(fpath):
    return str(fpath.parents[2].name).split('_')[0].lower()


def convert_and_clean_shapefile(fpath, out_dir=out_dir):
    df = gpd.read_file(fpath)
    # Drop unnecessary columns
    df.drop(columns=DROP_COLUMNS, inplace=True)
    df.rename(columns=RENAME_COLUMNS, inplace=True)

    # Add region name to each ESI segment to make them unique
    region_name = get_region(fpath)
    new_esi_id = [f'{region_name}-{id}' for id in df.esi_id]
    df['esi_id'] = new_esi_id

    # Write as new file
    out_file = out_dir / f'{region_name}-cleaned.geojson'
    logging.info(f'Writing {out_file}')
    df.to_file(out_file, driver='GeoJSON')


for fpath in esil_files:
    logging.info(f'Reading {fpath}')
    convert_and_clean_shapefile(fpath, out_dir)

# Load all cleaned files
geojson_files = out_dir.glob('*-cleaned.geojson')
dfs = [gpd.read_file(f) for f in geojson_files]
combined_gdf = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))

logging.info(f'Reading files from {base_dir}')

geojson_file = out_dir / 'combined-esi.geojson'
logging.info(f'Writing {geojson_file}')
combined_gdf.to_file(geojson_file, driver='GeoJSON')
logging.info(f'Writing {geojson_file}')
parquet_file = out_dir / 'combined-esi.parquet'
combined_gdf.to_parquet(parquet_file)
