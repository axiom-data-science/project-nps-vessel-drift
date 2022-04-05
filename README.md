# NPS - Vessel Drift Analysis

[![DOI](https://zenodo.org/badge/474192615.svg)](https://zenodo.org/badge/latestdoi/474192615)

## Summary

This repo is an archive of code, simulation results, and notebooks for the [Bering Sea Drift and Spill Vessel](https://spillanddrift.srv.axds.co) project to estimate the risk of oil spill to Alaskan shores from grounded vessels.

The goal of the project was to better understand the risks associated with vessel traffic in the Bering and Chukchi Seas to the Alaskan shoreline. This included estimating the probability of vessels drifting and running aground, estimating the likelihood of a grounded vessel spilling oil, and risk posed to coastal areas from oil spilled from such a vessel.

The project incorporated data from an industry database of 3000+ vessel incidents from Alaskan waters, a NOAA incident database, and a State of Alaska incident database to develop realistic parameters for the likelihood and duration of vessel drift events into drift simulation models.

## Repo Content

### Data

The results from the analysis are contained in two folders:

- `risk-files` contains Parquet files containing the oil spill hazard, oil spill risk, breadh hazard, organized Environmental Sensitivity Index (ESI) segment, vessel type, and month.
- `region-normalized-risk-files` contains the same fields as the `risk-files` folder, but normalized by the maximum value within the region.

### Code (src)

The scripts for processing the data for simulations, running, the simulations, and analyzing the simulations are contained
in the pip installable `vessel-drift-analysis` package.
