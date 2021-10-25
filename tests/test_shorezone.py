import numpy as np

from vessel_drift_analysis.shorezone import ShoreZone

SHOREZONE_PATH = '/mnt/store/data/assets/nps-vessel-spills/spatial-division/shorezone-shoretype.geojson'


class TestShoreZone:
    shorezone = ShoreZone(SHOREZONE_PATH)

    def test_shorezone_gdf(self):
        assert len(self.shorezone.gdf) == 194938
        assert all(self.shorezone.gdf.columns == ['bc_class', 'geometry'])

    def test_esi_loc(self):
        # Every point should have a bc_class != 0
        assert np.sum(self.shorezone.locs.bc_class == 0) == 0

        # bc_class codes must be in range [1, 39]
        assert np.array_equal(np.sort(self.shorezone.locs.bc_class.unique()), np.arange(1, 40))
