# Data container for AIS rasters used to launch drift simulations and analysis.
import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.spatial.ckdtree import cKDTree

VESSEL_TYPES = [
    'cargoShips',
    'passengerShips',
    'otherShips',
    'tankerShips'
]


class AISSet:
    """
    Interact with a set of AIS rasters used in vessel simulations.

    Arguments:
    ----------
    ais_dir: Path
        Path to directory with AIS tif files.
    year: int
        Year of AIS data to use.
    vessel_types: list
        List of vessel types as reflected in the AIS file names.

    Attributes:
    -----------
    ais_paths: list
        List of paths to each AIS file.
    """

    def __init__(self, ais_dir: Path, year: int, vessel_types: list = VESSEL_TYPES):
        self.dir = Path(ais_dir)
        self.vessel_types = vessel_types
        self.year = year
        self.paths = self._get_ais_paths()

    def _get_ais_paths(self) -> list:
        """Given path to directory with AIS data, return list of paths to files"""
        ais_files = []
        year = self.year
        end_year = self.year
        for month in range(1, 13):
            end_month = month + 1
            if month == 12:
                end_year += 1
                end_month = 1

            for vessel_type in self.vessel_types:
                path_template = f"{vessel_type}_{year}{month:02}01-{end_year}{end_month:02}01_total.tif"
                fname = self.dir / path_template
                ais_files.append(fname)

        return ais_files

    def get_ais_path(self, vessel_type: str, simulation_date: datetime.date) -> Path:
        """Given vessel_type and simulation date, return path to AIS raster"""
        # adjust date to match AIS file naming convention
        file_date = f"{self.year}{simulation_date.month:02}01"

        for path in self.paths:
            if vessel_type in path.name and file_date in path.name:
                break

        return path


class AIS:
    """
    Container for AIS raster data.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.date = self._get_date()
        self.vessel_counts = self._load_vessel_counts()
        locs = np.vstack((
            self.vessel_counts.lon.values,
            self.vessel_counts.lat.values
        )).T
        self.tree = cKDTree(locs)

    def _get_date(self) -> datetime.date:
        name = self.path.name
        # Assume name template: "{vessel_type}_{year}{month:02}01-{end_year}{end_month:02}01_total.tif"
        date = name.split('_')[1].split('-')[0]

        return datetime.datetime.strptime(date, '%Y%m%d')

    def _load_vessel_counts(self, crs: str = 'epsg:4326') -> gpd.GeoDataFrame:
        """Load vessel counts and return locations as GeoDataFrame"""
        with rasterio.open(self.path) as raster:
            vessel_counts = raster.read(1)

            # get locations where there were vessels (non-zero counts)
            non_zero_ix = np.argwhere(vessel_counts > 0)
            lon, lat = raster.xy(non_zero_ix[:, 0], non_zero_ix[:, 1])
            counts = vessel_counts[non_zero_ix[:, 0], non_zero_ix[:, 1]]

            gdf = gpd.GeoDataFrame(
                {
                    'counts': counts,
                    'lon': lon,
                    'lat': lat
                },
                geometry=gpd.points_from_xy(lon, lat)
            )
            return gdf.set_crs(crs)
