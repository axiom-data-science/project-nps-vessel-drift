#!/bin/bash
# Runs simulations and creates post-processed products

# 1. Launch drift sims
launch_drift.py -n 1000 -r 25000 -a /mnt/store/data/assets/nps-vessel-spills/ais-data/ais-data-2015-2020/processed_25km/2019/epsg4326 > drift_sim.log

# 2. Extract stranding locations
extract-stranding-locations.py drift-results stranding-locations > extract_strandings.log

# 3. Launch oil spill simulations from stranding locations
launch_oilspills.py -n 1000 -r 25000 cargo > cargo_spill.log &
launch_oilspills.py -n 1000 -r 25000 tanker > tanker_spill.log &
launch_oilspills.py -n 1000 -r 25000 passenger > passenger_spill.log &
launch_oilspills.py -n 1000 -r 25000 other > other_spill.log

wait

# 4. Calculate drift hazard
calculate_drift_hazard.py \
  --results_dir drift-results \
  --ais_dir ../../ais-data/ais-data-2015-2020/processed_25km/2019/epsg4326 \
  --esi_path ../../spatial-division/esi/cleaned-and-combined/combined-esi.parquet \
  --shorezone_path ../../spatial-division/shorezone-shoretype.geojson \
  --out_dir . > drift_hazard.log

# 5. Calculate spill hazard
calculate_spill_hazard.py \
  oil-spill-results \
  ../../spatial-division/esi/cleaned-and-combined/combined-esi.parquet \
  . > spill_hazard.log

# 6. Calculate total hazard and risk (by simulation date)
calculate-total-hazard-and-risk.py \
  drift_hazard.parquet \
  oil_spill_hazard.parquet \
  ../../spatial-division/esi/cleaned-and-combined/combined-esi.parquet \
  . > total_hazard.log

# 7. Calculate monthly hazard and risk
create-monthly-hazard-and-risk-files.py total-hazard monthly-files > mothly_files.log

# 8. Create portal files (non-normalized)
create-portal-files.py monthly-files  ../../spatial-division/esi/cleaned-and-combined/combined-esi.parquet portal-files > portal_files.log

# 9. Create GRS normalized portal files
create-grs-normalized-portal-files.py portal-files grs-normalized-portal-files > normalized_portal_files.log