# Convenience functions for working with the data
import numpy as np
import xarray as xr


def lon360_to_lon180(lon: np.ndarray) -> np.ndarray:
    """Given lon in [0, 360) range, return lon in [-180, 180)"""
    return np.mod(lon - 180, 360) - 180


def get_stranded_flag_from_status(ds: xr.Dataset) -> int:
    """Given Dataset of results, return int indicating stranded"""
    flag_meanings = ds.status.flag_meanings.split()
    for ix, flag in enumerate(flag_meanings):
        if flag == "stranded":
            return ix
