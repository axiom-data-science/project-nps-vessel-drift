import datetime

import numpy as np

import drift_results
from ais import AIS
from esi import ESI
from test_ais import AIS_FILE
from test_esi import ESI_PATH

SAMPLE_FILE = '/mnt/store/data/assets/nps-vessel-spills/results/50km_100v/alaska_drift_2019-01-17.nc'
NPARTICLES = 10300
NSTRANDED = 1886


class TestDriftResults:
    ais = AIS(AIS_FILE)
    esi = ESI(ESI_PATH)
    drift_result = drift_results.DriftResult(SAMPLE_FILE, ais, esi)

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
        assert stranded_per_esi_segment.nstranded.min() >= 0
        assert stranded_per_esi_segment.nstranded.max() <= NSTRANDED
        assert stranded_per_esi_segment.nstranded.sum() == NSTRANDED

    def test_pb_per_segment(self):
        pb_s = self.drift_result._calc_pb_per_esi_segment(self.esi)
        assert pb_s.pb_s.min() >= 0.0
        assert pb_s.pb_s.max() <= 1.0

    def test_calc_pb_per_particle(self):
        pb_per_particle = self.drift_result._calc_pb_per_particle(self.esi)
        assert len(pb_per_particle) == NPARTICLES
        assert pb_per_particle.min() >= 0.0
        assert pb_per_particle.max() <= 1.0

    def test_init(self):
        assert len(self.drift_result.data) == NPARTICLES

        assert self.drift_result.data.pt.min() >= 0.0
        assert self.drift_result.data.pt.max() <= 1.0

        assert self.drift_result.data.pb.min() >= 0.0
        assert self.drift_result.data.pb.max() <= 1.0

        assert self.drift_result.data.stranding_hazard.min() >= 0.0
        assert self.drift_result.data.stranding_hazard.max() <= 1.0

    def test_sim_start_date(self):
        assert self.drift_result.start_date == datetime.date(2019, 1, 17)