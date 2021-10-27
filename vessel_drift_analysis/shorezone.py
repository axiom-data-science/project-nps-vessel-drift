# Data container for Shorezone data
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class ShoreZone:
    """
    Shorezone data container.

    Attributes:
    -----------
    path: Path
        Path to Shorezone data.
    gdf: geopandas.GeoDataFrame
        Shorezone data.
    locs: pandas.DataFrame
        Points along every Shorezone segment including shorezone classification.
    tree: scipy.spatial.cKDTree
        Tree to query closest Shorezone point to look up shorezone classification.
    """

    def __init__(self, fpath: Path):
        self.path = fpath

        self.gdf = gpd.read_file(fpath)
        # Need tree + location lookup because gpd.query only looks over overlapping features
        # - Get (lon, lat) of every point in the geometry column to make a tree
        self.locs = shorezone_to_locs(self.gdf)
        self.tree = cKDTree(np.vstack((self.locs.lon.values, self.locs.lat.values)).T)

    def get_breach_prob(self, query_points: np.ndarray) -> np.ndarray:
        """Return probability of breaching based on Shorezone classification.

        Parameters:
        -----------
        query_points: np.ndarray
            (N, 2) array (lon, lat) of points to query shorezone classification for.

        Returns:
        --------
        np.ndarray
            (N,) array of probabilities of breaching.

        Notes:
        ------
        The probabilities for breaching derived from NOAA incident news from 2005-2015 [1]_.
        The probability of a vessel breaching on rocky coast is 0.7.
        The probability of a vessel breaching on non-rocky coast is 0.44.

        ..[1] https://incidentnews.noaa.gov
        """
        # Get Shorezone classification (bc_class) for each point
        _, ix = self.tree.query(query_points)
        bc_classes = self.locs.iloc[ix].bc_class.values

        # Convert stranding locations to probabilities that vessel will breach
        breach_probs = np.zeros((len(bc_classes),))
        breach_probs[bc_classes <= 20] = 0.7
        breach_probs[bc_classes > 20] = 0.44

        return breach_probs


def get_shorezone_npoints(shorezone: gpd.GeoDataFrame) -> int:
    """Return number of points in a Shorezone GeoDataFrame.

    Parameters:
    -----------
    shorezone: geopandas.GeoDataFrame
        Shorezone data.

    Returns:
    --------
    npoints: int
        Number of points in the Shorezone data.
    """
    count = 0
    for _, row in shorezone.iterrows():
        # row[0] - Shorezone bc_class
        # row[1] - Shorezone geometry
        npoints = len(np.array(row.geometry.xy).T)
        count += npoints

    return count


def shorezone_to_locs(shorezone: gpd.GeoDataFrame) -> pd.DataFrame:
    """Given Shorezone GeoDataFrame, return DataFrame of points with Shorezone classification.

    Parameters:
    -----------
    shorezone: geopandas.GeoDataFrame
        Shorezone data.

    Returns:
    --------
    locs: pandas.DataFrame
        Points and classification of each point in Shorezone data.

    Notes:
    ------
    DataFrame is needed as a lookup table for Shorezone classification when doing nearest-neighbor lookups.
    """
    shorezone_exploded = shorezone.explode()
    npoints = get_shorezone_npoints(shorezone_exploded)

    # lon, lat, bc_class
    lons = np.zeros((npoints,), dtype='f4')
    lats = np.zeros((npoints,), dtype='f4')
    bc_class = np.zeros((npoints,), dtype='i4')
    shorezone_rows = np.zeros((npoints,), dtype='i4')

    start_ix = 0
    for ix, row in shorezone_exploded.iterrows():
        # x, y = row[1] and transpose to be (n, 2)
        line_locs = np.array(row.geometry.xy).T
        nline_locs = len(line_locs)
        end_ix = start_ix + nline_locs

        lons[start_ix:end_ix] = line_locs[:, 0]
        lats[start_ix:end_ix] = line_locs[:, 1]
        bc_class[start_ix:end_ix] = row.bc_class
        # Knowing the row number in the original DataFrame is useful for look-ups
        shorezone_rows[start_ix:end_ix] = int(ix[0])

        start_ix = end_ix

    # return as a DataFrame
    df = pd.DataFrame(
        {
            'lon': lons,
            'lat': lats,
            'bc_class': bc_class,
            'esi_row': shorezone_rows
        }
    )
    return df
