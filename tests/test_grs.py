import numpy as np

from grs import GRS

GRS_PATH = "/mnt/store/data/assets/nps-vessel-spills/spatial-division/grs/grs.parquet"


class TestGRS:
    grs = GRS(GRS_PATH)

    def test_grs_gdf(self):
        assert len(self.grs.gdf) == 10

    def test_grs_loc(self):
        # Every point should have a GRS code
        assert np.sum(self.grs.locs.grs_code == 0) == 0

        # There should only be 10 codes
        assert len(np.unique(self.grs.locs.grs_code)) == 10

    def test_grs_tree(self):
        # Point in Homer should be in GRS Region Cook Inlet (404)
        # - Also, pattern for looking up points
        homer = np.array((-151.5483333, 59.6425))
        _, homer_grs_ix = self.grs.tree.query(homer)
        homer_grs = self.grs.locs.iloc[homer_grs_ix].grs_code
        assert homer_grs == 404
