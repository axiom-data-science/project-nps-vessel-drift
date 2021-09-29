# Data container for ESI data
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .grs import GRS


class ESI:
    """
    ESI data container.

    Attributes:
    -----------
    path: Path
        Path to ESI data
    gdf: geopandas.GeoDataFrame
        ESI data
    locs: pandas.DataFrame
        Points along every ESI segment including segment identifier (esi_id) and ESI code (esi_code)
    tree: scipy.spatial.cKDTree
        Tree to query closest ESI point to look up ESI segment identifier and ESI code
    """
    def __init__(self, fpath: Path):
        self.path = fpath

        self.gdf = gpd.read_parquet(self.path)
        # Need tree + location lookup because gpd.query only looks over overlapping features
        # - Get (lon, lat) of every point in the geometry column to make a tree
        self.locs = esi_to_locs(self.gdf)
        self.tree = cKDTree(np.vstack((self.locs.lon.values, self.locs.lat.values)).T)

    def get_grs_region_for_each_row(self, grs: GRS) -> np.ndarray:
        """Given GRS data container, return GRS code for each row in ESI data as array"""
        grs_codes_for_each_esi_row = self._get_grs_intersects(grs)
        grs_codes_for_each_esi_row = self._fill_grs_nonintersects(grs_codes_for_each_esi_row, grs)

        return grs_codes_for_each_esi_row

    def _get_grs_intersects(self, grs: GRS, grs_code_column_name: str = 'OBJECTID') -> np.ndarray:
        """Given GRS data container, return ndarray of GRS codes for each row in ESI data.

        Notes:
        - Filled with -9999 which is used as a missing value flag to identify rows that do not intersect
        """
        nrows = len(self.gdf)
        esi_to_grs_region = np.ones((nrows,)) * -9999

        for ix, row in self.gdf.iterrows():
            for grs_region in grs.gdf.index:
                if row.geometry.intersects(grs.loc[grs_region, 'geometry']):
                    esi_to_grs_region[ix] = grs.loc[grs_region, grs_code_column_name]
                break

        return esi_to_grs_region

    def _fill_grs_nonintersects(self, grs_codes_for_each_esi_row: np.ndarray, grs: GRS) -> np.ndarray:
        """Given array of GRS code for each ESI row, fill in missing GRS code using nearest neighbor.

        Notes:
        - Many points in ESI data do not intersect the GRS shape files
        - The points missing a GRS code (flagged as -9999) use a nearest neighbor lookup to fill in
        """
        esi_rows_missing_grs_ix = np.argwhere(grs_codes_for_each_esi_row == -9999)

        for missing_ix in esi_rows_missing_grs_ix:
            # There must be nicer syntax? But, this is robust
            x = self.gdf.iloc[esi_rows_missing_grs_ix[0]].geometry.values[0].geoms[0].xy[0][0]
            y = self.gdf.iloc[esi_rows_missing_grs_ix[0]].geometry.values[0].geoms[0].xy[1][0]
            _, grs_locs_ix = grs.grs_tree.query((x, y))
            missing_grs_code = grs.grs_locs[grs_locs_ix, 2]
            grs_codes_for_each_esi_row[missing_ix] = missing_grs_code

        return grs_codes_for_each_esi_row

    def __str__(self):
        return f'ESI for {self.path}'

    def __repr__(self):
        return f'ESI for {self.path}'


def get_npoints(gdf: gpd.GeoDataFrame) -> int:
    """Given exploded ESI, return the number of points in the file.

    Notes:
    - Can't extract this from the GDF because the points are exploded and encoded as linestrings.
    """
    count = 0
    for _, row in gdf.iterrows():
        # row[0] - ORI
        # row[1] - geometry which is LineString.  len is number of points
        npoints = len(np.array(row.geometry.xy).T)
        count += npoints

    return count


def clean_esi_code(esi_column: pd.Series) -> np.ndarray:
    """Given column of ESI codes, clean values, remove letters and return as integer array."""
    cleaned_esi_codes = np.zeros(len(esi_column), dtype='i2')
    for i, row in esi_column.iterrows():
        cleaned_esi_codes[i] = clean_esi_string(row)

    return cleaned_esi_codes


def esi_to_locs(esi: gpd.GeoDataFrame) -> pd.DataFrame:
    """Given ESI GeoDataFrame, return DataFrame of points with ESI codes and ids.

    Notes:
    - Array is returned needed for look-ups
    -
    """
    esi_exploded = esi.explode()
    npoints = get_npoints(esi_exploded)

    # lon, lat, esilgen_, esilgen_id, esi value, esi row in dataframe
    lons = np.zeros((npoints,), dtype='f4')
    lats = np.zeros((npoints,), dtype='f4')
    # max string length is 15 characters
    esi_ids = np.empty((npoints,), dtype='U15')
    esi_codes = np.zeros((npoints,), dtype='i4')
    esi_rows = np.zeros((npoints,), dtype='i4')

    # Iterate over each row
    # - Extract x, y points from each point in the line
    start_ix = 0
    for ix, row in esi_exploded.iterrows():            
        # x,y = row[1] and transpose to be (n, 2)
        line_locs = np.array(row.geometry.xy).T
        # number of points in the line
        nline_locs = len(line_locs)
        end_ix = start_ix + nline_locs

        lons[start_ix:end_ix] = line_locs[:, 0]
        lats[start_ix:end_ix] = line_locs[:, 1]
        esi_ids[start_ix:end_ix] = row.esi_id
        esi_codes[start_ix:end_ix] = clean_esi_string(row.esi)
        # Knowing the row number in the original DataFrame is useful for look-ups
        esi_rows[start_ix:end_ix] = int(ix[0])

        start_ix = end_ix

    # return as a dataframe
    df = pd.DataFrame(
        {
            'lon': lons,
            'lat': lats,
            'esi_id': pd.Series(esi_ids, dtype='U15'),
            'esi_code': pd.Series(esi_codes, dtype='i4'),
            'esi_row': pd.Series(esi_rows, dtype='i4')
        }
    )
    return df


def clean_esi_string(esi):
    """Given ESI string (e.g. from shapefile), return a single numeric value.

    Notes:
    - If no code is given, it is filled with the middle values 5 (on scale from 1 - 10)
    - Some codes have letters appended to end of number to distinguish sub-types, these are stripped
    - If there are more than 1 code, the high value is returned
    """
    codes = str(esi).split('/')

    esi_vals = []
    for code in codes:
        # if there is no code: give 5 as medium
        if code == 'None':
            code = 5
        # remove letter at end of string
        elif not code.isnumeric():
            code = code[:-1]
        esi_vals.append(int(code))

    # return the higest sensitivity
    return max(esi_vals)
