from vessel_drift_analysis import utils


def test_lon360_to_lon180():
    assert utils.lon360_to_lon180(200) == -160
    assert utils.lon360_to_lon180(180) == -180
    assert utils.lon360_to_lon180(360) == 0

