from datetime import date 
from pathlib import Path

from ais import AIS

AIS_PATH = '/mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2010-2013/processed/rescaled_25km_sum/wgs84/total/' 
AIS_FILE = '/mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2010-2013/processed/rescaled_25km_sum/wgs84/total/tankerShips_20120101-20120201_total.tif'
YEAR = 2012


class TestAIS:
    ais = AIS(Path(AIS_PATH), YEAR)

    def test_ais_paths(self):
        # Will test attributes:
        # - dir
        # - vessel_types
        # - year
        # There should be a AIS file for each vessel type for each month
        assert len(self.ais.paths) == len(self.ais.vessel_types) * 12

    def test_get_path(self):
        # Make sure date of simulation is different than date of AIS file
        test_date = date(2017, 1, 1)
        vessel_type = 'tankerShips'
        path = self.ais.get_ais_path(vessel_type, test_date)
        assert str(path) == AIS_FILE
