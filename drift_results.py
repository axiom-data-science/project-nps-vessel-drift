# Container for drift result simulations
import calendar
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import ais
import esi
import utils


class DriftResult:
    """
    Container for a drift result simulation.

    Arguments:
    ----------
    path: Path
        - Path to result file
    """
    def __init__(self, path: Path):#, ais_raster: ais.AIS, esi_raster: esi.ESI):
        self.path = path
#        self.pt_per_particle = self._calc_pt_per_particle(ais_raster)
#        self.pb_per_esi_segment = self._calc_pb_per_esi_segment(esi_raster)

    def _get_starting_points(self, crs: str = 'epsg:4326', convert_lon: bool = True) -> pd.DataFrame:
        """Return starting points for simulation as GeoDataFrame.

        Notes:
        - Converts lon from [0, 360] to [-180, 180] if convert_lon is True
        """
        with xr.open_dataset(self.path) as ds:
            ds0 = ds.isel(time=0)

        df = ds0.to_dataframe()

        # Aleutian project uses [0, 360) instead of [-180, 180) to avoid dateline issues
        # - Convert back to [-180, 180)
        if convert_lon:
            df.lon = utils.lon360_to_lon180(df.lon)

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.lon, df.lat)
        )
        return gdf.set_crs(crs)

    def _calc_pt_per_particle(self, ais: ais.AIS, **kwargs) -> np.ndarray:
        """Return Pt_r (probability of vessel at release point r) for each particle."""
        # Get starting position of very particle (drifting vessel)
        starting_points = self._get_starting_points(**kwargs)
        locs = np.vstack((starting_points.lon.values, starting_points.lat.values)).T

        # Find vessel count in AIS data from starting positing
        _, ix = ais.tree.query(locs)
        starting_counts = ais.vessel_counts.iloc[ix].counts.values

        # Pt is probability that a vessel is at the release point for the month.
        # - If there were more vessels than days of the month, make Pt = 1
        ndays_in_month = calendar.monthrange(ais.date.year, ais.date.month)[1]
        pt = starting_counts / ndays_in_month
        pt[pt > 1] = 1

        return pt

    def _get_stranded_per_esi_segment(self, esi: esi.ESI, **kwargs) -> defaultdict:
        """Return stranded vessels per ESI segment s."""
        esi_ids = self._get_esi_per_particle(esi, **kwargs)

        counts = defaultdict(int)
        for id in esi_ids:
            if id == '':
                continue
            counts[id] += 1

        return counts

    def _calc_pb_per_esi_segment(self, esi: esi.ESI, **kwargs) -> np.ndarray:
        """Return Pb_s, probability vessel drifted and stranded on ESI segment s."""
        stranded_by_esi = self._get_stranded_per_esi_segment(esi, **kwargs)
        stranded_counts = np.array([v for _, v in stranded_by_esi.items()])
        total_stranded = stranded_counts.sum()

        return stranded_counts / total_stranded

    def _get_esi_per_particle(
        self,
        esi: esi.ESI,
        dtype: str = 'U15',
        convert_lon: bool = True
    ) -> np.ndarray:
        """Return ESI segment for every vessel (only a segment ID for vessels that are stranded)."""
        with xr.open_dataset(self.path) as ds:
            stranded_flag = get_stranded_flag(ds)
            nvessels = len(ds.trajectory)
            esi_id_per_particle = np.empty(nvessels, dtype=dtype)

            # Get indices in dataset of where vessels are stranded
            stranded = ds.status.values == stranded_flag
            stranded_ix = np.argwhere(stranded)
            vessel_ix = stranded_ix[:, 0]
            time_ix = stranded_ix[:, 1]

            # Get stranding locations from dataset using indices
            lons = ds.lon.values[vessel_ix, time_ix]
            if convert_lon:
                lons = utils.lon360_to_lon180(lons)
            lats = ds.lat.values[vessel_ix, time_ix]
            locs = np.vstack((lons, lats)).T

        # Find ESI segment id using stranding locations
        _, ix = esi.tree.query(locs)
        esi_id_per_particle[vessel_ix] = esi.locs.iloc[ix].esi_id.values

        return esi_id_per_particle


def get_stranded_flag(ds):
    """Given DriftResult xr.Dataset, find the flag indicating stranded (not static between simulations)."""
    flag_meanings = ds.status.flag_meanings.split(' ')
    for ix, flag_meaning in enumerate(flag_meanings):
        if flag_meaning == 'stranded':
            return ix
