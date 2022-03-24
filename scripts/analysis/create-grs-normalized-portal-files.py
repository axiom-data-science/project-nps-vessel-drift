#!python
"""Create files with risk / hazard normalized by GRS region."""
import logging
from pathlib import Path

import geopandas as gpd

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


def normalize_by_grs(fpath: Path, output_dir: Path) -> gpd.GeoDataFrame:
    """Given path to portal results file, return GeoDataFrame with values normalized by GRS region."""
    gdf = gpd.read_parquet(fpath)
    regions = gdf.region.unique()
    for region in regions:
        logging.info(f'Processing region {region}')
        region_df = gdf.query(f'region=="{region}"')

        max_breach_hazard = region_df.breach_hazard.max()
        max_spill_hazard = region_df.breach_hazard.max()
        max_spill_risk = region_df.breach_hazard.max()

        region_df.breach_hazard = region_df.breach_hazard / max_breach_hazard
        region_df.spill_hazard = region_df.spill_hazard / max_spill_hazard
        region_df.spill_risk = region_df.spill_risk / max_spill_risk

        out_path = output_dir / fpath.name
        region_df.to_parquet(out_path)


def main(input_dir: Path, output_dir: Path):
    """Write GRS normalized files from input_dir to output_dir."""
    files = input_dir.glob('combined*.parquet')
    for f in files:
        if 'all' in str(f):
            continue
        logging.info(f'Processing {f}')
        normalize_by_grs(f, output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Path to directory with portal files'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Path to output directory'
    )
    args = parser.parse_args()
    logging.info(f'Reading files from {args.input_dir} and writing to {args.output_dir}')
    main(args.input_dir, args.output_dir)
