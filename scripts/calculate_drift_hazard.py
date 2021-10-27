#!python
# Script to calculate hazard to coasts by drifitng vessels.
import logging
from pathlib import Path

import pandas as pd

from vessel_drift_analysis.ais import AISSet
from vessel_drift_analysis.drift_results import (
    DriftResultsSet,
    get_vessel_type
)
from vessel_drift_analysis.esi import ESI

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s ', level=logging.INFO)


def calculate_hazard(
    result_set: DriftResultsSet,
    ais: AISSet,
    esi: ESI,
    **kwargs
):
    """Return combined drift result factors for a vessel type from simulation set."""
    # Find set of vessel types from result path names given
    vessel_types = {get_vessel_type(path) for path in result_set.paths}

    results = []
    for vessel_type in vessel_types:
        logging.info(f'- Loading results from {vessel_type} vessels')
        results.append(result_set.load_results(vessel_type, ais, esi, **kwargs))

    return pd.concat(results, ignore_index=True)


def main(
    results_dir: Path,
    ais_dir: Path,
    esi_path: Path,
    out_dir: Path,
    ais_year=2019,
):
    """Calculate drift hazard for all vessel types."""
    result_set = DriftResultsSet(results_dir)
    ais_set = AISSet(ais_dir, ais_year)
    esi = ESI(esi_path)

    hazard = calculate_hazard(result_set, ais_set, esi)
    out_path = out_dir / 'drift_hazard.parquet'
    logging.info(f'- writing results to {out_path}')
    hazard.to_parquet(out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate drift hazard')
    parser.add_argument('--results_dir', type=Path, required=True)
    parser.add_argument('--ais_dir', type=Path, required=True)
    parser.add_argument('--esi_path', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    args = parser.parse_args()

    logging.info('Calculating drift hazard')
    logging.info(f'- Drift results read from {args.results_dir.resolve()}')
    logging.info(f'- AIS data read from {args.ais_dir.resolve()}')
    logging.info(f'- ESI data read from {args.esi_path.resolve()}')
    logging.info(f'- Writing output to {args.out_dir.resolve()}')

    main(args.results_dir, args.ais_dir, args.esi_path, args.out_dir)
