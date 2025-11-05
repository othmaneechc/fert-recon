# Wheat 2002: check yield ∩ fertilizers (no parquet involved)
python -m sanity_check.check_source_coverage \
  --yield_nc /data/oe23/fert-recon/data/GlobalCropYield5min/GlobalCropYield5min1982_2015_V2.nc/Wheat1982_2015.nc \
  --fert_dir /data/oe23/fert-recon/data/Cropland_Maps \
  --mask_tif /data/oe23/fert-recon/exports/masks/Morocco_land_mask.tif \
  --crop wheat \
  --year 2002 \
  --out_png /data/oe23/fert-recon/exports/viz/coverage_wheat_2002.png

# Same, but also factor in monthly availability for March (month=3)
python -m sanity_check.check_source_coverage \
  --yield_nc /data/oe23/fert-recon/data/GlobalCropYield5min/GlobalCropYield5min1982_2015_V2.nc/Wheat1982_2015.nc \
  --fert_dir /data/oe23/fert-recon/data/Cropland_Maps \
  --mask_tif /data/oe23/fert-recon/exports/masks/Morocco_land_mask.tif \
  --monthly_dir /data/oe23/fert-recon/exports/monthly \
  --month 3 \
  --crop wheat \
  --year 2002 \
  --out_png /data/oe23/fert-recon/exports/viz/coverage_wheat_2002_m3.png

# Wheat (example year 2010)
python -m sanity_check.visualize_sources_global \
  --yield_nc /data/oe23/fert-recon/data/GlobalCropYield5min/GlobalCropYield5min1982_2015_V2.nc/Wheat1982_2015.nc \
  --fert_dir /data/oe23/fert-recon/data/Cropland_Maps \
  --crop wheat \
  --year 2010 \
  --out_dir /data/oe23/fert-recon/exports/viz_global

# Maize (example year 2010)
python -m sanity_check.visualize_sources_global \
  --yield_nc /data/oe23/fert-recon/data/GlobalCropYield5min/GlobalCropYield5min1982_2015_V2.nc/Maize1982_2015.nc \
  --fert_dir /data/oe23/fert-recon/data/Cropland_Maps \
  --crop maize \
  --year 2010 \
  --out_dir /data/oe23/fert-recon/exports/viz_global

