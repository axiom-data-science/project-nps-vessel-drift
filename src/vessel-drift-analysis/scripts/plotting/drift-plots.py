#!python
# Plot results from drift simulations.
import logging
from pathlib import Path

from vessel_drift_analysis import plot


def main(input_dir: Path, output_dir: Path) -> None:
    """Plot results from drift simulations.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing the simulation results.
    output_dir : str
        Path to the directory where the plots will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Plotting results from {input_dir} and saving to {output_dir}')

    input_files = input_dir.glob('*.nc')
    for input_file in input_files:
        plot.plot_results(input_file, output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot results from drift simulations.'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Path to the directory containing the simulation results.'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Path to the directory where the plots will be saved.'
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
