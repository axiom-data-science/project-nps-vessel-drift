#!python
# Rename raw AIS files
import logging
from pathlib import Path


def rename_file(path: Path) -> None:
    """Rename given AIS file.

    given: ais-heatmap-stage1.2_voyages_products_nps_satellite_2021_20210923T123234_alaska_eez_ALLShips_20201201-20210101_unique_500m.tif
    return: all_20201201-20210101_unique_500m.tif
    """
    *_, vessel_type, date, bin_type, resolution_ext = path.name.split('_')
    new_vessel_type = vessel_type.replace('Ships', '').lower()
    new_name = f'{new_vessel_type}_{date}_{bin_type}_{resolution_ext}'
    path.rename(new_name)


def rename_dir(dir_path: Path) -> None:
    """Rename all AIS files in given directory.
    """
    logging.info(f'Renaming files in {dir_path}')
    for file in dir_path.glob('*.tif'):
        if file.is_file():
            try:
                rename_file(file)
            # skip files that don't match expected naming convention, or other problems
            except:  # noqa
                continue


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rename AIS files')
    parser.add_argument('dir', type=Path, help='Directory to rename files')
    args = parser.parse_args()
    rename_dir(args.dir)
