import numpy as np

from vessel_drift_analysis.drift import calculate_drift_probability


def test_drift_probability():
    # Manually caluclated values
    REFERENCE_PROBABILITY = 0.0006849533841310851
    assert np.isclose(calculate_drift_probability(), REFERENCE_PROBABILITY)
