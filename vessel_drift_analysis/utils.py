# Convenience functions for working with the data
import numpy as np


def lon360_to_lon180(lon: np.ndarray) -> np.ndarray:
    """Given lon in [0, 360) range, return lon in [-180, 180)"""
    return np.mod(lon - 180, 360) - 180
