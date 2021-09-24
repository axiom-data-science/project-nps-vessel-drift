from datetime import date
from pathlib import Path

import numpy as np

from ais import AIS, AISSet

AIS_PATH = '/mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2010-2013/processed/rescaled_25km_sum/wgs84/total/'
AIS_FILE = '/mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2010-2013/processed/rescaled_25km_sum/wgs84/total/tankerShips_20120101-20120201_total.tif'
YEAR = 2012


class TestAISSet:
    ais_set = AISSet(AIS_PATH, YEAR)

    def test_ais_paths(self):
        # Will test attributes:
        # - dir
        # - vessel_types
        # - year
        # There should be a AIS file for each vessel type for each month
        assert len(self.ais_set.paths) == len(self.ais_set.vessel_types) * 12

    def test_get_path(self):
        # Make sure date of simulation is different than date of AIS file
        test_date = date(2017, 1, 1)
        vessel_type = 'tankerShips'
        path = self.ais_set.get_ais_path(vessel_type, test_date)
        assert str(path) == AIS_FILE


class TestAIS:
    ais = AIS(AIS_FILE)

    def test_ais_counts(self):
        # 216 pixels in example with vessel count greater than 0
        assert len(self.ais.vessel_counts) == 216

    def test_ais_tree(self):
        # Make sure tree is not empty
        assert len(self.ais.tree.data) > 0
        # Point in Homer should be in GRS Region Cook Inlet (404)
        # - Also, pattern for looking up points
        homer = np.array((-151.5483333, 59.6425))
        _, homer_ix = self.ais.tree.query(homer)
        assert self.ais.vessel_counts.iloc[homer_ix].counts == 255
