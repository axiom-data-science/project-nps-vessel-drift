# Container for drift result simulations
import calendar
from collections import defaultdict
import datetime
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ais import AIS
from esi import ESI
import utils


class DriftResult:
    """
    Container for a results of a single vessel drift simulation.

    Parameters
    ----------
    path: Path
        Path to result file
    ais: AIS
        AIS data container object.
    esi: ESI
        ESI data container object.
    prob_drift: float
        Probability of vessel drifting based on Vessels of Concern and AIS data from 2015-2019.

    Attributes
    ----------
    path: Path
        Path to simulation result file.
    start_date: datetime.date
        Date of the start of the simulation.
    data: pandas.DataFrame
        DataFrame of pt, pb, stranding_hazard, esi_id, and region indexed by particle number.
    """
    def __init__(self, path: Union[Path, str], ais: AIS, esi: ESI, prob_drift: float = 0.0005, **kwargs):
        self.path = Path(path)
        self.start_date = self._get_sim_starting_date()
        self.data = self._calc_drift_hazard(ais, esi, prob_drift, **kwargs)

    def to_parquet(self, path: Union[Path, str], **kwargs):
        """Write drift result to parquet file."""
        self.data.to_parquet(path, **kwargs)

    def _get_sim_starting_date(self) -> datetime.date:
        """Return simulation starting time as datetime.date"""
        # Use first time step in file to ensure we get the correct starting time
        with xr.open_dataset(self.path) as ds:
            date = ds.time[0].dt.date.data.item()

        return date

    def _calc_drift_hazard(self, ais: AIS, esi: ESI, prob_drift: float, **kwargs) -> pd.DataFrame:
        """Return drift hazard for each particle."""
        # Probability of vessel at release point r (Pt_r)
        pt = self._calc_pt_per_particle(ais, **kwargs)

        # Probability of vessel drifting and stranded on some ESI segment s (Pb_s)
        pb = self._calc_pb_per_particle(esi, **kwargs)

        # Probability of vessel drifting and stranding
        stranding_hazard = pt * pb * prob_drift

        # Add fields useful for grouping in analysis
        esi_per_particle = self._get_esi_per_particle(esi, **kwargs)
        # Change empty ESI IDs from '' to None
        esi_per_particle[esi_per_particle == ''] = None

        # ESI IDs are <region>-<segment #>, so we break the region out for convenience
        # - loop to deal with non-stranding ESI IDs
        region_per_particle = []
        for particle_esi_id in esi_per_particle:
            if particle_esi_id is None:
                region_per_particle.append(None)
            else:
                region_per_particle.append(particle_esi_id.split('-')[0])

        # Return all factors and stranding risk in single DataFrame
        return pd.DataFrame(
            {
                'pt': pt,
                'pb': pb,
                'stranding_hazard': stranding_hazard,
                'esi_id': esi_per_particle,
                'region': region_per_particle
            },
            index=np.arange(len(pt))
        )

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

    def _calc_pt_per_particle(self, ais: AIS, **kwargs) -> np.ndarray:
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

    def _get_stranded_per_esi_segment(self, esi: ESI, **kwargs) -> pd.DataFrame:
        """Return stranded vessels per ESI segment s."""
        esi_ids = self._get_esi_per_particle(esi, **kwargs)

        counts = defaultdict(int)
        for id in esi_ids:
            if id == '':
                continue
            counts[id] += 1

        return pd.DataFrame({'nstranded': counts.values()}, index=counts.keys())

    def _calc_pb_per_esi_segment(self, esi: ESI, **kwargs) -> pd.DataFrame:
        """Return Pb_s, probability vessel drifted and stranded on ESI segment s."""
        stranded_by_esi = self._get_stranded_per_esi_segment(esi, **kwargs)
        pb_s = stranded_by_esi / stranded_by_esi.sum()
        pb_s.rename(columns={'nstranded': 'pb_s'}, inplace=True)

        return pb_s

    def _calc_pb_per_particle(self, esi: ESI, **kwargs) -> np.ndarray:
        """Return Pb per particle."""
        # Array of segments IDs per particle ('' if not stranded)
        esi_per_particle = self._get_esi_per_particle(esi, **kwargs)
        # DataFrame of pb_s (indexed by esi_id)
        pb_per_segment = self._calc_pb_per_esi_segment(esi, **kwargs)
        # Map esi_per_particle to pb_per_segment
        # - Unable to use esi_per_particle because it includes empty ESI IDs ('') for non-stranded particles
        # - Add a row for non-stranded particles with pb_s = 0
        pb_non_stranded = pd.DataFrame({'pb_s': 0}, index=[''])
        pb_per_segment_patched = pb_per_segment.append(pb_non_stranded)
        pb_per_particle = np.squeeze(pb_per_segment_patched.loc[esi_per_particle].values)

        return pb_per_particle

    def _get_esi_per_particle(
        self,
        esi: ESI,
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


def get_stranded_flag(ds: xr.Dataset) -> int:
    """Given DriftResult xr.Dataset, find the flag indicating stranded (not static between simulations)."""
    flag_meanings = ds.status.flag_meanings.split(' ')
    for ix, flag_meaning in enumerate(flag_meanings):
        if flag_meaning == 'stranded':
            return ix
