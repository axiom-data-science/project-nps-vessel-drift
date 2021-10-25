#!python
"""Launch vessel drift simulations.

Methodology
-----------
For every week with forcing data, simulated oil spills are launched from location where
a simulated vessel ran aground in the vessel drift simulations.

Oil spills are assumed to be marine diesel oil with each spill consisting of 1000 elements.
The amount spilled is dependending on the vessel type with values derived from the Washington
State Department of Ecology and Environment (WSDEE)[1].

The oil spill model ("Adios") is parameterized to use dispersion, evaporation, and emulsification.
Vertical mixing is turned off for these simulations.  All oil is released instantaneously.


References
----------
[1] https://www.wsdee.wa.gov/programs/vessel-drift/vessel-drift-data-and-model-parameters
"""
import datetime
import logging
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from opendrift.models.openoil import OpenOil
from opendrift.readers import (
    reader_netCDF_CF_generic,
    reader_shape
)

logging.basicConfig(level=logging.WARNING)


@dataclass
class SimulationConfig:
    """Configuration for a single OpenDrift simulation"""
    start_date: datetime.datetime
    readers: List
    number: int
    radius: float = 1000  # this is meters from given x,y point
    time_step: int = 900
    time_step_output: int = 86400
    duration: datetime.timedelta = datetime.timedelta(days=7)
    outfile: str = None
    loglevel: int = logging.INFO
    grounding_dir: Path = Path('/mnt/store/data/assets/nps-vessel-spills/stranding-locations/satellite')


# Values drived from: https://apps.ecology.wa.gov/publications/documents/96250.pdf
# - Guidance on amount of oil by vessel types, but focused values typical for Puget Sound.
# - Broadly applicable for this purpose, however.
# - Worst case scenarios numbers
SpillConfig = namedtuple('spill_config', ['type', 'amount'])
GAL_TO_M3 = 0.003785
OIL_CONFIGS = {
    'cargo': SpillConfig('MARINE INTERMEDIATE FUEL OIL', 2_000_000 * GAL_TO_M3),
    # The vast majority of "other" are fishing vessels, so we'll just assume that here
    'other': SpillConfig('MARINE INTERMEDIATE FUEL OIL', 75_000 * GAL_TO_M3),
    'passenger': SpillConfig('MARINE INTERMEDIATE FUEL OIL', 2_000_000 * GAL_TO_M3),
    'tanker': SpillConfig('MARINE INTERMEDIATE FUEL OIL', 5_000_000 * GAL_TO_M3)
}

# vessel types from AIS
VESSEL_TYPES = [
    'tanker',
    'passenger',
    'other',
    'cargo'
]


# ~11 min per test
def run_sim(run_config, oil_configs, vessel_type):
    logging.info(f'oil spill simulation started for {run_config.start_date:%Y-%m-%d}')
    start_time = time.perf_counter()

    # load vessel grounding locations for this week
    grounding_dir = run_config.grounding_dir
    grounding_path = grounding_dir / f'{vessel_type}_alaska_drift_{run_config.start_date:%Y-%m-%d}.npy'
    locs = np.load(grounding_path)
    lons = locs[:, 0]
    lats = locs[:, 1]

    # prep and launch the simulation
    oil_type = oil_configs[vessel_type].type
    oil_amount = oil_configs[vessel_type].amount
    oil_sim = OpenOil(loglevel=run_config.loglevel, weathering_model='noaa')
    oil_sim.add_reader(run_config.readers)
    oil_sim.seed_elements(
        lon=lons,
        lat=lats,
        # Single time indicates release of all oil at one time
        time=run_config.start_date,
        number=len(lons),
        oiltype=oil_type,
        m3_per_hour=oil_amount
    )
    oil_sim.set_config('general:use_auto_landmask', False)  # Disabling the automatic GSHHG landmask
    oil_sim.set_config('processes:dispersion', True)
    oil_sim.set_config('processes:evaporation', True)
    oil_sim.set_config('processes:emulsification', True)
    oil_sim.set_config('drift:vertical_mixing', False)
    # Add default values for readers that are not provided
    oil_sim.set_config('environment:fallback:upward_sea_water_velocity', 0)
    oil_sim.set_config('environment:fallback:sea_surface_wave_significant_height', 0)
    oil_sim.set_config('environment:fallback:sea_surface_wave_stokes_drift_x_velocity', 0)
    oil_sim.set_config('environment:fallback:sea_surface_wave_stokes_drift_y_velocity', 0)
    oil_sim.set_config('environment:fallback:sea_surface_wave_period_at_variance_spectral_density_maximum', 0)
    oil_sim.set_config('environment:fallback:sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment', 0)  # noqa
    oil_sim.set_config('environment:fallback:sea_ice_area_fraction', 0)
    oil_sim.set_config('environment:fallback:sea_ice_x_velocity', 0)
    oil_sim.set_config('environment:fallback:sea_ice_y_velocity', 0)
    oil_sim.set_config('environment:fallback:sea_water_temperature', 5)
    # changed input from "sea_water_practical_salinity" to "sea_water_salinity"
    oil_sim.set_config('environment:fallback:sea_water_salinity', 34)
    # changed input from "depth" to "sea_floor_depth_below_sea_level"
    oil_sim.set_config('environment:fallback:sea_floor_depth_below_sea_level', 50)
    oil_sim.set_config('environment:fallback:ocean_vertical_diffusivity', 0.1)

    oil_sim.run(
        time_step=run_config.time_step,
        time_step_output=run_config.time_step_output,
        duration=run_config.duration,
        outfile=run_config.outfile
    )

    end_time = time.perf_counter()
    total_time = int(end_time - start_time)
    logging.info(f'simulation complete {total_time} s')

    return total_time


def run_simulations(
    days=7,
    number=100,
    radius=1000,
    timestep=900,
    output_timestep=3600,
    vessel_types=VESSEL_TYPES,
    oil_configs=OIL_CONFIGS,
    loglevel=logging.INFO
):
    if type(vessel_types) is str:
        vessel_types = [vessel_types]

    # start date possible to launch drifter, limited by availability of HYCOM data
    start_date = datetime.datetime(2019, 1, 17)
    # last date possible to launch drifter, limited by availability of NAM data
    last_date = datetime.datetime(2019, 12, 10)
    date = start_date
    duration = datetime.timedelta(days=days)

    # currents
    hycom_file = '/mnt/store/data/assets/nps-vessel-spills/forcing-files/hycom/updated-files/hycom.nc'
    hycom_reader = reader_netCDF_CF_generic.Reader(hycom_file)

    # winds
    fname = '/mnt/store/data/assets/nps-vessel-spills/forcing-files/nam/regrid/nam.nc'
    nam_reader = reader_netCDF_CF_generic.Reader(fname)

    # land - cannot use default landmask as it is -180, 180
    # Instead, we use the same landmask with lons shifted to 0, 360
    fname = '/mnt/store/data/assets/nps-vessel-spills/sim-scripts/drift/world_0_360.shp'
    reader_landmask = reader_shape.Reader.from_shpfiles(fname)

    # Reader order matters.  first reader sets the projection for the simulation.
    readers = [hycom_reader, nam_reader, reader_landmask]

    sim_start_time = time.perf_counter()
    while date <= last_date:
        for vessel_type in vessel_types:
            try:
                logging.info(f'launching simulation for {vessel_type} starting on {date:%Y-%m-%d}')
                output_fname = f'oilspill_{vessel_type}_{date:%Y-%m-%d}.nc'
                config = SimulationConfig(
                    date,
                    readers,
                    number,
                    radius,
                    timestep,
                    output_timestep,
                    duration,
                    output_fname,
                    loglevel
                )
                run_sim(config, oil_configs, vessel_type)

            except Exception as e:
                logging.warning(f'simulation failed for {date:%Y-%m-%d}')
                logging.warning(str(e))

        date = date + datetime.timedelta(days=days)

    sim_end_time = time.perf_counter()
    total_sim_time = int(sim_end_time - sim_start_time)
    logging.info(f'total sim time {total_sim_time} s')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--number',
        default=1000,
        type=int,
        help='Number of vessels to launch per simulation'
    )
    parser.add_argument(
        '-r',
        '--radius',
        default=100,
        type=float,
        help='Max distance from release point to launch vessel (in meters)'
    )
    parser.add_argument(
        'vessel_type',
        type=str,
        help='vessel type ("cargo", "tanker", "passenger", "other")'
    )
    args = parser.parse_args()
    run_simulations(
        days=7,
        number=args.number,
        radius=args.radius,
        timestep=900,
        output_timestep=86400,
        vessel_types=args.vessel_type,
        loglevel=logging.INFO
    )


if __name__ == '__main__':
    main()
