#!python
"""Launch vessel drift simulations.

Methodology
-----------
For every week with forcing data, simulated vessels are launched from every cell where
a vessel was present in the AIS data.

The drift angle (up to 60 deg.), windage scaling (2% - 10% of 10 m), and left/right
direction is randomly assigned per simulated vessel.
"""
import datetime
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.models.oceandrift import LagrangianArray
from opendrift.readers import reader_netCDF_CF_generic, reader_shape
from rasterio import warp

logging.basicConfig(level=logging.WARNING)
RANGE_LIMIT_RADS = 60 * np.pi / 180
TIF_DIR = '/mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2015-2020/processed_25km/2019/epsg4326'


class Vessel(LagrangianArray):
    """Extend LagrangianArray for use with Alaskan Vessel Drift Project."""
    variables = LagrangianArray.add_variables([
        (
            'wind_scale',
            {
                'dtype': np.float32,
                'units': '1',
                'default': 1
            }
        ),
        (
            'wind_offset',
            {
                'dtype': np.float32,
                'units': '1',
                'default': 1
            }
        )
    ])


class AlaskaDrift(OpenDriftSimulation):
    ElementType = Vessel
    required_variables = [
        'x_wind',
        'y_wind',
        'eastward_sea_water_velocity',
        'northward_sea_water_velocity',
        'eastward_sea_ice_velocity',
        'northward_sea_ice_velocity',
        'sea_ice_area_fraction',
        'land_binary_mask'
    ]

    def seed_elements(
        self,
        lon,
        lat,
        radius=0,
        number=None,
        time=None,
        seed=187,
        range_limit_rads=RANGE_LIMIT_RADS,
        **kwargs
    ):
        if number is None:
            number = self.get_config('seed:number_of_elements')

        # drift is going to be a random value between 2% - 10% of wind
        # (b - a) * random_sample + a
        # a = 0.02
        # b = 0.1
        wind_scale = (0.1 - 0.02) * np.random.random_sample((number,)) + 0.02
        # offset is -60 deg. to 60 deg.
        # a = -60
        # b = 60
        # (60 - (-60)) * random_sample + (-60)
        wind_offset = (range_limit_rads + range_limit_rads) * np.random.random_sample((number,)) - range_limit_rads # noqa

        super(AlaskaDrift, self).seed_elements(
            lon=lon,
            lat=lat,
            radius=radius,
            number=number,
            time=time,
            wind_scale=wind_scale,
            wind_offset=wind_offset,
            **kwargs
        )

    def update(self):
        """Update ship position taking into account wind, currents, stokes, and ice."""
        # Inspired by `advect_oil`
        if hasattr(self.environment, 'sea_ice_area_fraction'):
            ice_area_fraction = self.environment.sea_ice_area_fraction
            # Above 70%–80% ice cover, the oil moves entirely with the ice.
            k_ice = (ice_area_fraction - 0.3) / (0.8 - 0.3)
            k_ice[ice_area_fraction < 0.3] = 0
            k_ice[ice_area_fraction > 0.8] = 1

            factor_stokes = (0.7 - ice_area_fraction) / 0.7
            factor_stokes[ice_area_fraction > 0.7] = 0
        else:
            k_ice = 0
            factor_stokes = 1

        # 1. update wind
        windspeed = np.sqrt(self.environment.x_wind**2 + self.environment.y_wind**2)
        windspeed *= self.elements.wind_scale

        # update angle using random offset +- 60 deg
        # windir is in rads, so need to convert
        winddir = np.arctan2(self.environment.y_wind, self.environment.x_wind)
        winddir += self.elements.wind_offset
        wind_x = windspeed * np.cos(winddir)
        wind_y = windspeed * np.sin(winddir)

        # Scale wind by ice factor
        wind_x = wind_x * (1 - k_ice)
        wind_y = wind_y * (1 - k_ice)
        self.update_positions(wind_x, wind_y)

        # 2. update with sea_water_velocity
        # This assumes x_sea_water_velocity and not eastward_sea_water_velocity...
        #self.advect_ocean_current(factor=1 - k_ice)
        self.update_positions(
            self.environment.eastward_sea_water_velocity * (1 - k_ice),
            self.environment.northward_sea_water_velocity * (1 - k_ice)
        )

        # 3. Advect with ice
        self.advect_with_sea_ice(factor=k_ice)

        # Deactivate elements that hit the land mask
        self.deactivate_elements(
            self.environment.land_binary_mask == 1,
            reason='ship stranded'
        )


@dataclass
class SimulationConfig:
    """Configuration for a single OpenDrift simulation"""
    start_date: datetime.datetime
    readers: List
    number: int
    radius: float = 25000  # this is meters from given x,y point
    time_step: int = 900
    time_step_output: int = 3600
    duration: datetime.timedelta = datetime.timedelta(days=7)
    outfile: str = None
    loglevel: int = logging.INFO


def lonlat_from_tif(date, tif_file, dst_crs=rasterio.crs.CRS.from_epsg(4326)):
    """Return (lon, lat) in TIFF with cell value > 0"""
    with rasterio.open(tif_file) as ds:
        src_crs = ds.crs
        idx = np.argwhere(ds.read(1))
        x, y = ds.xy(idx[:, 0], idx[:, 1])

        lon, lat = warp.transform(
            src_crs,
            dst_crs,
            x,
            y
        )
        # need to change from [-180, 180] to [0, 360]
        lon = np.array(lon) % 360
        lat = np.array(lat)

    return lon, lat


# ~2 min per test
def run_sims_for_date(run_config, tif_dir=TIF_DIR):
    vessel_types = ['cargo', 'other', 'passenger', 'tanker']

    # Run simulation using data for start date for every vessel type
    month = run_config.start_date.month

    tif_files = list(Path(tif_dir).glob('*.tif'))
    tif_files.sort()

    base_fname = run_config.outfile
    for vessel_type in vessel_types:
        try:
            tif_file = list(Path(tif_dir).glob(f'{vessel_type}_2019{month:02}01-2019*.tif'))[0]
        except IndexError:
            if month == 12:
                tif_file = list(Path(tif_dir).glob(f'{vessel_type}_2019{month:02}01-2020*.tif'))[0]
            else:
                raise IndexError(f"No AIS data found for {month}")

        logging.info(f'Starting simulation preparation for {tif_file=}')

        vessel_type = tif_file.name.split('.')[0].split('_')[0]
        # prepend out name with vessel type
        outfile = vessel_type + '_' + base_fname

        # release points from each ais location where a vessel was in the past
        lons, lats = lonlat_from_tif(run_config.start_date, tif_file)

        # launch vessel simulation
        vessel_sim = AlaskaDrift(loglevel=run_config.loglevel)
        vessel_sim.add_reader(run_config.readers)
        for i in range(run_config.number):
            vessel_sim.seed_elements(
                lon=lons,
                lat=lats,
                time=run_config.start_date,
                number=len(lons),
                radius=run_config.radius
            )
        # Disabling the automatic GSHHG landmask
        vessel_sim.set_config('general:use_auto_landmask', False)

        # Backup velocities
        vessel_sim.set_config('environment:fallback:sea_ice_area_fraction', 0)
        vessel_sim.set_config('environment:fallback:northward_sea_ice_velocity', 0)
        vessel_sim.set_config('environment:fallback:eastward_sea_ice_velocity', 0)
        vessel_sim.set_config('environment:fallback:northward_sea_water_velocity', 0)
        vessel_sim.set_config('environment:fallback:eastward_sea_water_velocity', 0)
        vessel_sim.set_config('environment:fallback:x_wind', 0)
        vessel_sim.set_config('environment:fallback:y_wind', 0)
        vessel_sim.run(
            time_step=run_config.time_step,
            time_step_output=run_config.time_step_output,
            duration=run_config.duration,
            outfile=outfile
        )


def run_simulations(
    days=7,
    number=50,
    radius=5000,
    timestep=900,
    output_timestep=3600,
    tif_dir=TIF_DIR,
    loglevel=logging.INFO
):
    # start date possible to launch drifter, limited by availability of HYCOM data
    start_date = datetime.datetime(2019, 12, 5)
    # last date possible to launch drifter, limited by availability of NAM data
    last_date = datetime.datetime(2019, 12, 18)
    date = start_date
    duration = datetime.timedelta(days=days)

    # currents + ice
    hycom_file = '/mnt/store/data/assets/nps-vessel-spills/forcing-files/hycom/final-files/hycom.nc'
    # Provide a name mapping to work with package methods:
    name_map = {
        'eastward_sea_water_velocity': 'x_sea_water_velocity',
        'northward_sea_water_velocity': 'y_sea_water_velocity',
        'siu': 'x_sea_ice_velocity',
        'siv': 'y_sea_ice_velocity',
    }
    hycom_reader = reader_netCDF_CF_generic.Reader(hycom_file, standard_name_mapping=name_map)

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
        try:
            logging.info(f'simulation started for {date:%Y-%m-%d}')
            start_time = time.perf_counter()
            output_fname = f'alaska_drift_{date:%Y-%m-%d}.nc'
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
            run_sims_for_date(config, tif_dir)
            end_time = time.perf_counter()
            total_time = int(end_time - start_time)
            logging.info(f'simulation complete {total_time} s')
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
        default=50,
        type=int,
        help='Number of vessels to launch per simulation'
    )
    parser.add_argument(
        '-r',
        '--radius',
        default=25000,
        type=float,
        help='Max distance from release point to launch vessel (in meters)'
    )
    parser.add_argument(
        '-a',
        '--ais',
        default=TIF_DIR,
        type=str,
        help='Path to dir with AIS tifs for release points'
    )
    args = parser.parse_args()
    run_simulations(
        days=7,
        number=args.number,
        radius=args.radius,
        timestep=900,
        output_timestep=86400,
        tif_dir=args.ais,
        loglevel=logging.INFO
    )


if __name__ == '__main__':
    main()
