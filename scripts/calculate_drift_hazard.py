#!python
# Script to hazard to coasts by drifitng vessels.
from pathlib import Path
from typing import List

import pandas as pd

from vessel_drift_analysis.ais import VESSEL_TYPES, AISSet
from vessel_drift_analysis.drift_results import DriftResultsSet
from vessel_drift_analysis.esi import ESI


def calculate_hazard(
    result_set: DriftResultsSet,
    ais: AISSet,
    esi: ESI,
    vessel_types: List[str],
    **kwargs
):
    """Return combined drift result factors for a vessel type from simulation set."""
    results = []
    breakpoint()
    for vessel_type in vessel_types:
        results.append(result_set.load_results(vessel_type, ais, esi, **kwargs))

    return pd.concat(results, ignore_index=True)


def main(
    results_dir: Path,
    ais_dir: Path,
    esi_path: Path,
    out_dir: Path,
    ais_year=2013,
    vessel_types=VESSEL_TYPES
):
    """Calculate drift hazard for all vessel types."""
    result_set = DriftResultsSet(results_dir)
    ais_set = AISSet(ais_dir, ais_year)
    esi = ESI(esi_path)

    hazard = calculate_hazard(result_set, ais_set, esi, vessel_types)
    hazard.to_parquet(out_dir / 'drift_hazard.parquet')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate drift hazard')
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--ais_dir', type=Path, required=True)
    parser.add_argument('--esi_path', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    args = parser.parse_args()

    print('Calculating drift hazard')
    print(f'{args.results_dir=}')
    print(f'{args.ais_dir=}')
    print(f'{args.esi_path=}')
    print(f'{args.out_dir=}')

    main(args.results_dir, args.ais_dir, args.esi_path, args.out_dir)
