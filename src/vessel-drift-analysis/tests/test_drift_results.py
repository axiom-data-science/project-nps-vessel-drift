import datetime
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from vessel_drift_analysis import drift_results
from vessel_drift_analysis.ais import AIS, AISSet
from vessel_drift_analysis.esi import ESI
from vessel_drift_analysis.shorezone import ShoreZone

from test_ais import AIS_FILE, AIS_PATH, YEAR
from test_esi import ESI_PATH
from test_shorezone import SHOREZONE_PATH


SAMPLE_DIR = Path('/mnt/store/data/assets/nps-vessel-spills/sim-results/terrestial-sims/drift-sensitivity-analysis/50km_100v')  # noqa
SAMPLE_FILE = SAMPLE_DIR / 'alaska_drift_2019-01-17.nc'
# Not 52 because we don't have the forcing files to cover the entire year of 2019
NSAMPLE_FILES = 47
NPARTICLES = 10300
NSTRANDED = 1886

# Don't want to load these twice
global_ais = AIS(AIS_FILE)
global_esi = ESI(ESI_PATH)
global_shorezone = ShoreZone(SHOREZONE_PATH)


class TestDriftResultsSet:
    result_set = drift_results.DriftResultsSet(SAMPLE_DIR)
    ais_set = AISSet(AIS_PATH, YEAR)
    esi = global_esi
    shorezone = global_shorezone

    def test_paths(self):
        assert len(self.result_set.paths) == NSAMPLE_FILES

    def test_load_results(self):
        # 'alaska' not an actual vessel type, just use for testing
        tanker_results = self.result_set.load_results('alaska', self.ais_set, self.esi, self.shorezone)
        assert tanker_results.date.unique().size == NSAMPLE_FILES


class TestDriftResults:
    ais = global_ais
    esi = global_esi
    shorezone = global_shorezone
    drift_result = drift_results.DriftResult(SAMPLE_FILE, ais, esi, shorezone)

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

    def test_calc_breach_prob_per_particle(self):
        beach_prob_per_particle = self.drift_result._calc_breach_prob_per_particle(self.shorezone)
        assert len(beach_prob_per_particle) == NPARTICLES
        assert beach_prob_per_particle.min() >= 0.0
        assert beach_prob_per_particle.max() <= 1.0

    def test_init(self):
        assert len(self.drift_result.data) == NPARTICLES

        assert self.drift_result.data.pt.min() >= 0.0
        assert self.drift_result.data.pt.max() <= 1.0

        assert self.drift_result.data.pb.min() >= 0.0
        assert self.drift_result.data.pb.max() <= 1.0

        assert self.drift_result.data.stranding_hazard.min() >= 0.0
        assert self.drift_result.data.stranding_hazard.max() <= 1.0

        assert self.drift_result.data.breach_prob.min() >= 0.0
        assert self.drift_result.data.breach_prob.max() <= 1.0

    def test_sim_start_date(self):
        assert self.drift_result.start_date == datetime.date(2019, 1, 17)

    def test_to_parquet(self):
        try:
            _, fpath = tempfile.mkstemp(suffix='.parquet')
            self.drift_result.to_parquet(fpath)
            read_data = pd.read_parquet(fpath)
            assert self.drift_result.data.equals(read_data)
        finally:
            os.remove(fpath)
