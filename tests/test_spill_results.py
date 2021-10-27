import datetime
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from test_esi import ESI_PATH

from vessel_drift_analysis import spill_results
from vessel_drift_analysis.esi import ESI

SAMPLE_DIR = Path('/mnt/store/data/assets/nps-vessel-spills/sim-results/satellite-sims/oil-spill-results/')
SAMPLE_FILE = SAMPLE_DIR / 'oilspill_tanker_2019-12-05.nc'
NSAMPLE_FILES = 188
NPARTICLES = 12532
NSTRANDED = 11202
NESI_SEGMENTS = 2034
VESSEL_TYPE = 'tanker'


class TestSpillResultsSet:
    result_set = spill_results.SpillResultsSet(SAMPLE_DIR)
    esi = ESI(ESI_PATH)

    def test_paths(self):
        assert len(self.result_set.paths) == NSAMPLE_FILES

    def test_load_results(self):
        tanker_results = self.result_set.load_results('tanker', self.esi)
        # NSAMPLE_FILES = 188 comprised of the four vessel_types
        assert tanker_results.date.unique().size == NSAMPLE_FILES // 4


class TestSpillResults:
    esi = ESI(ESI_PATH)
    spill_result = spill_results.SpillResult(SAMPLE_FILE, esi, VESSEL_TYPE)

    def test_esi_id_per_particle(self):
        esi_ids = self.spill_result._get_esi_per_particle(self.esi)
        assert len(esi_ids) == NPARTICLES
        assert np.sum(esi_ids != '') == NSTRANDED

    def test_oil_mass_per_particle(self):
        oil_mass = self.spill_result._get_oil_mass_per_particle()
        assert len(oil_mass) == NPARTICLES

    def test_calc_concentration_index(self):
        concentration_index = self.spill_result._calc_concentration_index(self.esi)
        assert len(concentration_index) == NESI_SEGMENTS
        assert concentration_index.oil_mass.min() >= 0.0
        assert concentration_index.particle_hits.min() >= 0.0
        assert concentration_index.cs.min() >= 0.0
        assert concentration_index.cs.max() <= 1.0

    def test_init(self):
        assert len(self.spill_result.data) == NESI_SEGMENTS

        assert self.spill_result.data.oil_mass.min() >= 0.0

        assert self.spill_result.data.particle_hits.min() >= 0.0

        assert self.spill_result.data.pb.min() >= 0.0
        assert self.spill_result.data.pb.max() <= 1.0

        assert self.spill_result.data.cs.min() >= 0.0
        assert self.spill_result.data.cs.max() <= 1.0

    def test_sim_start_date(self):
        assert self.spill_result.start_date == datetime.date(2019, 12, 5)

    def test_to_parquet(self):
        try:
            _, fpath = tempfile.mkstemp(suffix='.parquet')
            self.spill_result.to_parquet(fpath)
            read_data = pd.read_parquet(fpath)
            assert self.spill_result.data.equals(read_data)
        finally:
            os.remove(fpath)
