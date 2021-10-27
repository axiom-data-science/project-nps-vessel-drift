import numpy as np

from vessel_drift_analysis.esi import ESI

ESI_PATH = "/mnt/store/data/assets/nps-vessel-spills/spatial-division/esi/cleaned-and-combined/combined-esi.parquet"  # noqa


class TestESI:
    esi = ESI(ESI_PATH)

    def test_esi_gdf(self):
        assert len(self.esi.gdf) == 104212
        assert all(self.esi.gdf.columns == ['esi_id', 'esi', 'geometry'])

    def test_esi_loc(self):
        # Every point should have an ESI code
        assert np.sum(self.esi.locs.esi_code == 0) == 0

        # ESI codes must be in range [1, 10]
        assert np.array_equal(np.sort(self.esi.locs.esi_code.unique()), np.arange(1, 11))
