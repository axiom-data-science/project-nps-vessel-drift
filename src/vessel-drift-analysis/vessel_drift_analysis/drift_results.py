# Container for drift result simulations
import calendar
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Union, DefaultDict

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from . import utils
from .ais import AIS, AISSet
from .esi import ESI
from .shorezone import ShoreZone


class DriftResult:
    """
    Container for a results of a single vessel drift simulation.

    Parameters
    ----------
    path: Path
        Path to result file.
    ais: AIS
        AIS data container object.
    esi: ESI
        ESI data container object.
    prob_drift: float
        Probability of vessel drifting based on Vessels of Concern and AIS data from
        2015-2019.

    Attributes
    ----------
    path: Path
        Path to simulation result file.
    start_date: datetime.date
        Date of the start of the simulation.
    data: pandas.DataFrame
        DataFrame of `pt`, `pb`, `stranding_hazard`, `esi_id`, and `region` indexed by particle number.

    Notes
    -----
    The terms `pt`, `pb`, and `stranding_hazard` are used to estimate the probability of
    a vessel drifting and becoming stranded on a particular section of interest along a
    coastline using the methodology adopted from [1]_, [2]_.

    `pt` is the probability that a vessel was at the release point based on AIS data.

    `pb` is ratio of vessels that drifted and stranded on an ESI segment to the total
    number of vessels for the simulation.

    `stranding_hazard` is the probability that a vessel drifted and stranded based on
    vessels of concern and AIS data from 2015-2019.

    Methods with names including `calc_` include actual calculations whereas `get_`
    indicates data subsetting.

    ..[1] Sepp Neves, A.A., Pinardi, N. & Martins, F. IT-OSRA: applying ensemble
    simulations to estimate the oil spill risk associated to operational and accidental
    oil spills. Ocean Dynamics 66, 939–954 (2016).
    https://doi.org/10.1007/s10236-016-0960-0

    ..[2] Sepp Neves, A.A., Pinardi, N., Martins F., Janeiro, J., Samaras, A., Zodiatis,
    G., De Dominicis, M., Towards a common oil spill risk assessment framework –
    Adapting ISO 31000 and addressing uncertainties, Journal of Environmental
    Management,Volume 159, 2015, Pages 158-168, ISSN 0301-4797,
    https://doi.org/10.1016/j.jenvman.2015.04.044.
    """
    def __init__(
        self,
        path: Union[Path, str],
        ais: AIS,
        esi: ESI,
        shorezone: ShoreZone,
        prob_drift: float = 0.0006,
        **kwargs
    ):
        self.path = Path(path)
        self.start_date = self._get_sim_starting_date()
        self.vessel_type = ais.vessel_type
        self.data = self._calc_drift_hazard(ais, esi, shorezone, prob_drift, **kwargs)

    def to_parquet(self, path: Union[Path, str], **kwargs) -> None:
        """Write drift result to parquet file.

        Parameters
        ----------
        path: Path
            Path to write parquet file to.
        """
        self.data.to_parquet(path, **kwargs)

    def _get_sim_starting_date(self) -> datetime.date:
        """Return simulation start date.

        Returns
        -------
        start_date: datetime.date
            Simulation start date.
        """
        # Use first time step in file to ensure we get the correct starting time
        with xr.open_dataset(self.path) as ds:
            date = ds.time[0].dt.date.data.item()

        return date

    def _calc_drift_hazard(
        self,
        ais: AIS,
        esi: ESI,
        shorezone: ShoreZone,
        prob_drift: float,
        **kwargs
    ) -> pd.DataFrame:
        """Return drift hazard for each particle.

        Parameters
        ----------
        ais: AIS
            AIS data container object.
        esi: ESI
            ESI data container object.
        shorezone: ShoreZone
            Shorezone data container object.
        prob_drift: float
            Probability of vessel drifting.

        Returns
        -------
        drift_hazard: pandas.DataFrame
            Terms and regions associated with drift hazard calculations on a per particle basis.
        """
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
        region_per_particle: list[Union[str, None]] = []
        for particle_esi_id in esi_per_particle:
            if particle_esi_id is None:
                region_per_particle.append(None)
            else:
                region_per_particle.append(particle_esi_id.split('-')[0])

        # Add probability that vessel will breach based on Shorezone data about coastline
        breach_prob = self._calc_breach_prob_per_particle(shorezone)

        # Return all factors, stranding risk, and breaching probability in a single DataFrame
        df = pd.DataFrame(
            {
                'pt': pt,
                'pb': pb,
                'stranding_hazard': stranding_hazard,
                'breach_prob': breach_prob,
                'esi_id': esi_per_particle,
                'region': region_per_particle,
            },
            index=np.arange(len(pt))
        )
        df.attrs.update({'start_date': self.start_date})

        return df

    def _get_starting_points(self, crs: str = 'epsg:4326', convert_lon: bool = True) -> gpd.GeoDataFrame:
        """Return drifting vessel starting points from simulation.

        Parameters
        ----------
        crs: str
            Coordinate reference system of simulation output. Typically 'epsg:4326'.
        convert_lon: bool
            Convert longitude values from 0 to 360 to -180 to 180.

        Returns
        -------
        starting_points: geopandas.GeoDataFrame
            GeoDataFrame of starting points for each particle in simulation indexed by particle number.
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

    def _calc_breach_prob_per_particle(
        self,
        shorezone: ShoreZone,
        convert_lon: bool = True,
    ) -> np.ndarray:
        """Return probability of a vessel breaching and spilling oil based on coastline data.

        Parameters
        ----------
        shorezone: ShoreZone
            Shorezone data container object.

        Returns
        -------
        breach_prob: np.ndarray
            Probability of breaching and spilling oil for each particle / vessel.
        """
        with xr.open_dataset(self.path) as ds:
            stranded_flag = utils.get_stranded_flag_from_status(ds)
            nvessels = len(ds.trajectory)
            breach_prob_per_particle = np.zeros((nvessels,))

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
        breach_prob_per_particle[vessel_ix] = shorezone.get_breach_prob(locs)

        return breach_prob_per_particle

    def _calc_pt_per_particle(self, ais: AIS, **kwargs) -> np.ndarray:
        """Return probability of vessel at release point at start of simulation (`pt`).

        Parameters
        ----------
        ais: AIS
            AIS data container object.

        Returns
        -------
        pt: np.ndarray
            Probability of vessel at release point at start of simulation.
        """
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
        """Return the number of stranded vessels per ESI segment.

        Parameters
        ----------
        esi: ESI
            ESI data container object.

        Returns
        -------
        stranded_vessels_per_esi_segment: pandas.DataFrame
            Number of stranded vessels per ESI segment.
        """
        esi_ids = self._get_esi_per_particle(esi, **kwargs)

        counts: DefaultDict[str, int] = defaultdict(int)
        for id in esi_ids:
            if id == '':
                continue
            counts[id] += 1

        return pd.DataFrame({'nstranded': counts.values()}, index=counts.keys())

    def _calc_pb_per_esi_segment(self, esi: ESI, **kwargs) -> pd.DataFrame:
        """Return probability vessel drifted and stranded on coastline.

        Parameters
        ----------
        esi: ESI
            ESI data container object.

        Returns
        -------
        pb: pandas.DataFrame
            Probability of vessel drifting and stranded on coastline indexed by ESI segment.
        """
        stranded_by_esi = self._get_stranded_per_esi_segment(esi, **kwargs)
        pb_s = stranded_by_esi / stranded_by_esi.sum()
        pb_s.rename(columns={'nstranded': 'pb_s'}, inplace=True)

        return pb_s

    def _calc_pb_per_particle(self, esi: ESI, **kwargs) -> np.ndarray:
        """Return `pb` of ESI segment where vessel stranded.

        Parameters
        ----------
        esi: ESI
            ESI data container object.

        Returns
        -------
        pb: np.ndarray
            `pb` of ESI segment where vessel stranded.
        """
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
        """Return ESI segment for each vessel.

        Parameters
        ----------
        esi: ESI
            ESI data container object.
        dtype: str
            Data type of ESI segment IDs. (Default: 'U15')
        convert_lon: bool
            Convert longitude values from 0 to 360 to -180 to 180. (Default: True)

        Returns
        -------
        esi_ids: np.ndarray
            ESI segment for each vessel.
        """
        with xr.open_dataset(self.path) as ds:
            stranded_flag = utils.get_stranded_flag_from_status(ds)
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


class DriftResultsSet:
    """Interact with a set of simulation results."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.dir = Path(path)
        if not self.dir.is_dir() and self.dir.suffix == '.nc':
            paths = [self.dir]
        elif self.dir.is_dir():
            paths = list(self.dir.glob('*.nc'))
        else:
            raise ValueError(f'{path} is not a directory or a .nc file.')
        self.paths = sorted(paths)

    def load_results(
        self,
        vessel_type: str,
        ais_set: AISSet,
        esi: ESI,
        shorezone: ShoreZone
    ) -> pd.DataFrame:
        """Load all available results."""
        vessel_specific_paths = [p for p in self.paths if p.name.startswith(vessel_type)]
        results = []
        for path in vessel_specific_paths:
            # load AIS data used to init this simulation
            start_date = get_sim_start_date(path)
            ais_path = ais_set.get_ais_path(vessel_type, start_date)
            ais = AIS(ais_path)

            tmp_results = DriftResult(path, ais, esi, shorezone)
            # add date as column to provide ability to group by date
            tmp_results.data['date'] = tmp_results.data.attrs['start_date']
            # vessel type is also useful when combining results from multiple vessel types
            tmp_results.data['vessel_type'] = vessel_type
            results.append(tmp_results)

        return pd.concat([r.data for r in results], ignore_index=True)


def get_vessel_type(drift_result_path: Path) -> str:
    """Given a Path to a drift simulation result, return the vessel type.

    Parameters
    ----------
    drift_result_path: Path
        Path to drift simulation results

    Returns
    -------
    vessel_type: str
        The vessel type

    Notes
    -----
    Assumes the output file name template is <vessel-type>_alaska_drift_<%Y-%m-%d>.nc
    """
    vessel_type, *_ = drift_result_path.name.split('.')[0].split('_')

    return vessel_type


def get_sim_start_date(drift_result_path: Path) -> datetime.date:
    """Given a Path to a drift simulation result, return the simulation start date.

    Parameters
    ----------
    drift_result_path: Path
        Path to drift simulation results

    Returns
    -------
    start_date: datetime.date
        Start date of simulation

    Notes
    -----
    Assumes the output file name template is <vessel-type>_alaska_drift_<%Y-%m-%d>.nc
    """
    *_, start_date = drift_result_path.name.split('.')[0].split('_')
    start_date = datetime.date.fromisoformat(start_date)

    return start_date
