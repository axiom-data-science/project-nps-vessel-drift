import numpy as np
from test_ais import AIS_FILE
from test_esi import ESI_PATH

import drift_results
from ais import AIS
from esi import ESI

SAMPLE_FILE = '/mnt/store/data/assets/nps-vessel-spills/results/50km_100v/alaska_drift_2019-01-17.nc'
NPARTICLES = 10300
NSTRANDED = 1886


class TestDriftResults:
    drift_result = drift_results.DriftResult(SAMPLE_FILE)
    ais = AIS(AIS_FILE)
    esi = ESI(ESI_PATH)

    def test_starting_points(self):
        df = self.drift_result._get_starting_points()
        assert len(df.lon.values) == NPARTICLES
        assert len(df.lat.values) == NPARTICLES

    def test_pt_per_particle(self):
        pt = self.drift_result._calc_pt_per_particle(self.ais)
        assert len(pt) == NPARTICLES
        assert max(pt) <= 1.0
        assert min(pt) >= 0.0

    def test_esi_id_per_particle(self):
        esi_ids = self.drift_result._get_esi_per_particle(self.esi)
        assert len(esi_ids) == NPARTICLES
        assert np.sum(esi_ids != '') == NSTRANDED

    def test_stranded_per_esi_segment(self):
        stranded_per_esi_segment = self.drift_result._get_stranded_per_esi_segment(self.esi)
        assert min(stranded_per_esi_segment.values()) >= 0
        assert max(stranded_per_esi_segment.values()) <= NSTRANDED
        assert sum(stranded_per_esi_segment.values()) == NSTRANDED

    def test_pb_per_segment(self):
        pb_s = self.drift_result._calc_pb_per_esi_segment(self.esi)
        assert pb_s.min() >= 0.0
        assert pb_s.max() <= 1.0
