#!python
"""Convert shp files to geojson, combined, and save as parquet."""
from pathlib import Path

import geopandas as gpd
import pandas as pd

RENAME_COLUMNS = {
    'ESILGEN_': 'esi_id',
    'ESI': 'esi'
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
    'LENGTH'
]

base_dir = Path('/mnt/store/data/assets/nps-vessel-spills/spatial-division/esi/esi-shapefiles')
esil_files = base_dir.glob('*/**/esil.shp')
out_dir = Path('/mnt/store/data/assets/nps-vessel-spills/spatial-division/esi/cleaned-geojson')
out_dir.mkdir(exist_ok=True)


def get_region(fpath):
    return str(fpath).split('/')[1].split('_')[0].lower()


def convert_and_clean_shapefile(fpath, out_dir=out_dir):
    df = gpd.read_file(fpath)
    df.drop(columns=DROP_COLUMNS, inplace=True)
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    region_name = get_region(fpath)
    new_esi_id = [f'{region_name}-{id}' for id in df.esi_id]
    df['esi_id'] = new_esi_id
    df.to_file(f'{region_name}-cleaned.geojson', driver='GeoJSON')


for fpath in esil_files:
    print(fpath)
    convert_and_clean_shapefile(fpath)

geojson_files = out_dir.glob('*.geojson')
dfs = [gpd.read_file(f) for f in geojson_files]
combined_gdf = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
combined_gdf.to_file('combined-esi.geojson', driver='GeoJSON')
combined_gdf.to_parquet('combined-esi.parquet')
