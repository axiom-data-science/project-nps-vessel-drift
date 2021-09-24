import drift_results

import ais
from test_ais import AIS_FILE

SAMPLE_FILE = '/mnt/store/data/assets/nps-vessel-spills/results/50km_100v/alaska_drift_2019-01-17.nc'
NPARTICLES = 10300


class TestDriftResults:
    drift_result = drift_results.DriftResult(SAMPLE_FILE)
    ais = ais.AIS(AIS_FILE)

    def test_starting_points(self):
        df = self.drift_result._get_starting_points()
        assert len(df.lon.values) == NPARTICLES
        assert len(df.lat.values) == NPARTICLES

    def test_pt(self):
        pt = self.drift_result._calc_pt(self.ais)
        assert len(pt) == NPARTICLES
        assert max(pt) <= 1.0
        assert min(pt) >= 0.0
