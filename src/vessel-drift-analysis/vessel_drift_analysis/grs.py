# Data container for GRS data
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class GRS:
    """
    GRS data container.

    Attributes:
    -----------
    path: Path
        Path to GRS shapefile
    gdf: geopandas.GeoDataFrame
        GRS data
    locs: geopandas.GeoDataFrame
        Points along GRS region boundaries with GRS code
    tree: scipy.spatial.cKDTree
        Tree to query closest GRS point to look up GRS region
    """
    def __init__(self, fpath: Path):
        self.path = fpath

        self.gdf = gpd.read_parquet(self.path)
        self.locs = grs_to_locs(self.gdf)
        # cKDTree expects a numpy array of shape (n, 2)
        self.tree = cKDTree(np.vstack((self.locs.lon.values, self.locs.lat.values)).T)

        # Useful for analysis
        self.gdf.set_index('NAME', inplace=True)

    def __repr__(self):
        return f'GRS for {self.path}'

    def __str__(self):
        return f'GRS for {self.path}'


def get_grs_npoints(exploded_gdf) -> int:
    """Given exploded GRS GeoDataFrame, return total number of points in all lines.

    Notes:
    - Useful for allocating array for grs_to_locs and lookup tables
    """
    count = 0
    for _, row in exploded_gdf.iterrows():
        # row[0] - ORI
        # row[1] - geometry which is LineString.  len is number of points in string
        count += len(row.geometry.boundary.xy[0])

    return count


def grs_to_locs(grs_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Given GRS GeoDataFrame, return DataFrame for each point along GRS boundary with GRS code."""
    # GRS geometry is comprised of MultiPolygons.  We want to seperate those to Polygons.
    grs_exploded = grs_gdf.explode()

    # allocate array to hold GRS code ("OBJECTID") for every point
    npoints = get_grs_npoints(grs_exploded)
    # (lon, lat, grs_code)
    locs = np.zeros((npoints, 3), dtype='f4')

    # Iterate over each row
    # - Extract x, y points from the boundary of each Polygon
    start_ix = 0
    end_ix = 0
    for ix, row in grs_exploded.iterrows():
        # x, y = row[1] and transpose to be (n, 2)
        line_locs = np.array(row.geometry.boundary.xy).T
        # number of points in this line
        nline_locs = len(line_locs)

        # objectid is int encoding of the region
        objectid = np.ones((nline_locs,)) * row[0]

        end_ix = start_ix + nline_locs
        locs[start_ix:end_ix, :2] = line_locs
        locs[start_ix:end_ix, 2] = objectid
        start_ix = end_ix

    df = pd.DataFrame(
        {
            'lon': locs[:, 0],
            'lat': locs[:, 1],
            'grs_code': pd.Series(locs[:, 2], dtype='int32')
        }
    )

    return df
