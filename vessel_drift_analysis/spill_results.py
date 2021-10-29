# Container for spill result simulations
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from . import utils
from .esi import ESI


class SpillResult:
    """
    Container for a results of a single vessel drift simulation.

    Parameters
    ----------
    path: Path
        Path to result file.
    esi: ESI
        ESI data container object.

    Attributes
    ----------
    path: Path
        Path to simulation result file.
    start_date: datetime.date
        Date of the start of the simulation.
    data: pandas.DataFrame
        DataFrame of `oil_mass`, `cs`, `pb`, `particle_hits`, `esi_id`, `region`, `date`, and `vessel_type`
        grouped by, but not indexed, `esi_id`.

    Notes
    -----
    The term `cs` is used to estimate the hazard of a vessel drifting, running aground, then
    breaching, spilling oil, and inundating the coastline using the methodology adopted from
    [1]_, [2]_.  The values for `cs` are saved per particle, but are calculated and valid for every
    particle that landed in the same ESI segment.  That is, if two particles (A and B) land in
    ESI segment `west-001`, then the value of `cs` for particle A is the same as the value of `cs`
    for particle B.

    `beached_oil` is the amount of the oil spilled and beached per particle.

    `cs` the concentration index of the beached oil spilled.  It is the sum of oil spilled in
    each ESI segment divided by the total amount of oil spilled in the domain.

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
    def __init__(self, path: Union[Path, str], esi: ESI, vessel_type: str, **kwargs):
        self.path = Path(path)
        self.start_date = self._get_sim_starting_date()
        self.vessel_type = vessel_type
        self.data = self._calc_concentration_index(esi, **kwargs)

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

    def _calc_concentration_index(self, esi: ESI, **kwargs) -> pd.DataFrame:
        """Return concentration index per esi segment.

        Parameters
        ----------
        esi: ESI
            ESI data container object.

        Returns
        -------
        oil_mass: pandas.DataFrame
            Beached oil mass per esi segment, indexed by ESI segment.

        Notes
        -----
        All terms returned are aggregated values over ESI segments.
        """
        oil_mass = self._get_oil_mass_per_particle()
        esi_ids = self._get_esi_per_particle(esi, **kwargs)

        oil_mass_by_esi = defaultdict(float)
        particles_by_esi = defaultdict(int)
        region_by_esi = defaultdict(str)
        for mass, id in zip(oil_mass, esi_ids):
            if id == '':
                continue
            # Get sum of mass of oil spilled in each ESI segment
            oil_mass_by_esi[id] += mass
            # Count all particles that landed in each ESI segment
            particles_by_esi[id] += 1
            # Get GRS region of each ESI segment
            region_by_esi[id] = id.split('-')[0]

        # From Sepp-Neves (2016):
        # "Cs is the concentration index, defined as the ensemble mean concentration
        # of beached oil at each coastal site normalized by the maximum mean concentration
        # value found in the domain."
        # Here we are using mass of oil, the concentration  (mass / area?) can be computed later.
        # We really have a length of coastline, not an area...
        # Will use ESI segment length to normalize this.  It will be mass / length (kg / km).
        oil_mass = np.fromiter(oil_mass_by_esi.values(), dtype=float)

        # Pb_s - Probability spill hit each ESI segment
        particle_count_per_esi = np.fromiter(particles_by_esi.values(), dtype=int)
        pb = particle_count_per_esi / particle_count_per_esi.sum()
        ensemble_mean_mass = oil_mass / particle_count_per_esi
        cs = ensemble_mean_mass / oil_mass.max()

        df = pd.DataFrame(
            {
                'oil_mass': oil_mass_by_esi.values(),
                'cs': cs,
                'pb': pb,
                'particle_hits': particle_count_per_esi,
                'esi_id': oil_mass_by_esi.keys(),
                'region': region_by_esi.values()
            },
            index=np.arange(len(cs))
        )
        df.attrs.update({'start_date': self.start_date})

        return df

    def _get_oil_mass_per_particle(
        self,
    ) -> np.ndarray:
        """Return the mass of oil per particle.

        Parameters
        ----------
        esi: ESI
            ESI data container object.

        Returns
        -------
        mass_oil: np.ndarray
            Mass of oil per particle.
        """
        with xr.open_dataset(self.path) as ds:
            # Get indices in dataset of where particles beached / stranded
            stranded_flag = utils.get_stranded_flag_from_status(ds)
            stranded = ds.status.values == stranded_flag
            stranded_ix = np.argwhere(stranded)
            particle_ix = stranded_ix[:, 0]
            time_ix = stranded_ix[:, 1]

            nvessels = len(ds.trajectory)
            oil_mass_per_particle = np.empty(nvessels)
            oil_mass_per_particle[particle_ix] = ds.mass_oil.values[particle_ix, time_ix]

        return oil_mass_per_particle

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


class SpillResultsSet:
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

    def load_results(self, vessel_type: str, esi: ESI) -> pd.DataFrame:
        """Load all available results."""
        spill_specific_paths = [p for p in self.paths if vessel_type in str(p.name)]
        results = []
        for path in spill_specific_paths:
            tmp_results = SpillResult(path, esi, vessel_type)
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
    Assumes the output file name template is oilspill_<vessel-type>_<%Y-%m-%d>.nc
    """
    _, vessel_type, *_ = drift_result_path.name.split('.')[0].split('_')

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
    Assumes the output file name template is oilspill_<vessel-type>_<%Y-%m-%d>.nc
    """
    *_, start_date = drift_result_path.name.split('.')[0].split('_')
    start_date = datetime.date.fromisoformat(start_date)

    return start_date
