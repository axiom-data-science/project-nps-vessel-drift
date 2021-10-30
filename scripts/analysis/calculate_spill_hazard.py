#!python
# Script to calculate the spill hazard for a given vessel
import logging
from pathlib import Path

import pandas as pd

from vessel_drift_analysis.esi import ESI
from vessel_drift_analysis.spill_results import SpillResultsSet, get_vessel_type

logging.basicConfig(format='%(process)d - %(levelname)s: %(message)s', level=logging.INFO)


def calculate_hazard(
    result_set: SpillResultsSet,
    esi: ESI,
    **kwargs
):
    """Return combined spill result factors for calculating hazard from a simulation set."""
    vessel_types = {get_vessel_type(path) for path in result_set.paths}

    results = []
    for vessel_type in vessel_types:
        logging.info(f'- Loading results from {vessel_type} vessels')
        results.append(result_set.load_results(vessel_type, esi))

    return pd.concat(results, ignore_index=True)


def main(
    results_dir: Path,
    esi_path: Path,
    out_dir: Path,
):
    """Calculate oil spill hazard for all vessel types."""
    result_set = SpillResultsSet(results_dir)
    esi = ESI(esi_path)

    hazard = calculate_hazard(result_set, esi)
    out_path = out_dir / 'oil_spill_hazard.parquet'
    logging.info(f'- writing results to {out_path}')
    hazard.to_parquet(out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate drift hazard')
    parser.add_argument('results_dir', type=Path)
    parser.add_argument('esi_path', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()

    logging.info('Calculating drift hazard')
    logging.info(f'- Drift results read from {args.results_dir.resolve()}')
    logging.info(f'- ESI data read from {args.esi_path.resolve()}')
    logging.info(f'- Writing output to {args.out_dir.resolve()}')

    main(args.results_dir, args.esi_path, args.out_dir)
