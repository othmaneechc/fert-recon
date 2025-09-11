#!/usr/bin/env bash
###############################################################################
# pipeline.sh
#
# End-to-end workflow:
#   0. configure → COUNTRY / YEARS / DATASETS
#   1. call cli.py to (re)export all Google-Earth-Engine rasters
#      • SoilGrids       (static, country-wide)
#      • Per-year region (MODIS, TerraClimate, CHIRPS, …)
#   2. call extract_all.py
#      • unzips every export → output/processed/tiles/*.tif
#      • guarantees ONE band per file
#      • creates yearly multi-band stacks in output/processed/stack_yearly/
###############################################################################
set -euo pipefail

# ─────────────────────────── USER SETTINGS ──────────────────────────────
COUNTRY="Morocco"               # passed to cli.py
YEARS=($(seq 2000 2015))        # change as needed
WORKERS=16                      # if your cli.py supports parallel exports

# per-year image collections
REGION_DS=(
  modis_ndvi_evi    # NDVI & EVI stats
  modis_lai_fapar   # LAI & fAPAR stats
  modis_lst         # Day & Night LST stats
  modis_et          # ET stats
  chirps            # CHIRPS rainfall stats
  era5              # ERA5 temperature stats
  terraclimate      # 86-variable TerraClimate stats
  srtm_slope        # SRTM elevation / slope
  jrc_gsw           # 7 Global Surface-Water layers
  modis_mcd12q1     # 17 MODIS land-cover classes
)

# ─────────────────────────── EXPORT STEP ───────────────────────────────
echo
echo "===== 1. Google-Earth-Engine EXPORT ====="

## 1.a SoilGrids (country-wide, static)
echo
echo "→ SoilGrids ($COUNTRY)"
python3 data-pipeline/scripts/cli.py soilgrids \
    --country "$COUNTRY" \
    --out "output/SoilGrids"

## 1.b Region‐based composites (one ZIP per statistic)
for YEAR in "${YEARS[@]}"; do
  for DS in "${REGION_DS[@]}"; do
    OUT="output/region/${DS}/${YEAR}"
    mkdir -p "$OUT"
    echo
    echo "→ $DS  •  $COUNTRY  •  $YEAR → $OUT"
    python3 data-pipeline/scripts/cli.py region "$DS" \
        --country "$COUNTRY" \
        --year    "$YEAR" \
        --out     "$OUT" 
  done
done

echo
echo "🎉  All EE exports finished"

# ─────────────────────────── EXTRACTION & STACKING ─────────────────────
echo
echo "===== 2. Unzip → Tiles → Yearly stacks ====="
python3 data-pipeline/scripts/extract_all.py

echo
echo "🎉  Pipeline done!"
echo "    Single-band tiles  : output/processed/tiles/"
echo "    Yearly mega-stacks : output/processed/stack_yearly/"
