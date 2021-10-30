#!python
"""Create files for portal ingestion from total hazard files."""
import logging
from pathlib import Path

import geopandas as gpd


logging.basicConfig(format='%(process)d - %(levelname)s: %(message)s', level=logging.INFO)


# Steps:
# - Load all the data
# - For every month:
#   - Load data available for that month
#   - Create hazard and risk for the month using each simulation as a weighted average
#   - Write result in parquet