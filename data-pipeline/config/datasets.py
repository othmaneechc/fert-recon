# config/datasets.py
DATASETS = {
    # CHIRPS: use DAILY and aggregate to monthly sum (mm/month)
    "chirps_precip": {
        "type": "collection",
        "ee_id": "UCSB-CHG/CHIRPS/DAILY",
        "band": "precipitation",
        "cadence": "DAILY",
        "agg": "sum",
        "scale_m": 5000,
        "description": "CHIRPS daily precipitation aggregated to monthly sum (mm)."
    },

    # ERA5-Land monthly aggregates (official public ID)
    "era5_temp2m": {
        "type": "collection",
        "ee_id": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "band": "temperature_2m",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 11132,
        "description": "ERA5-Land monthly aggregated 2m air temperature (K)."
    },
    "era5_soil_moisture": {
        "type": "collection",
        "ee_id": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "band": "volumetric_soil_water_layer_1",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 11132,
        "description": "ERA5-Land monthly aggregated soil moisture layer 1 (m3/m3)."
    },

    # MODIS vegetation indices (monthly product)
    "modis_ndvi": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD13A3",
        "band": "NDVI",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS monthly NDVI (scale 1e-4)."
    },
    "modis_evi": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD13A3",
        "band": "EVI",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS monthly EVI (scale 1e-4)."
    },

    # MODIS LST (8-day) → monthly mean
    "modis_lst_day": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD11A2",
        "band": "LST_Day_1km",
        "cadence": "8D",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS 8-day LST Day aggregated to monthly mean (×0.02 K)."
    },
    "modis_lst_night": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD11A2",
        "band": "LST_Night_1km",
        "cadence": "8D",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS 8-day LST Night aggregated to monthly mean (×0.02 K)."
    },

    # # MODIS ET/PET (8-day) → monthly sum
    # "modis_et": {
    #     "type": "collection",
    #     "ee_id": "MODIS/061/MOD16A2",
    #     "band": "ET",
    #     "cadence": "8D",
    #     "agg": "sum",
    #     "scale_m": 500,
    #     "description": "MODIS 8-day ET aggregated to monthly sum (×0.1 mm)."
    # },
    # "modis_pet": {
    #     "type": "collection",
    #     "ee_id": "MODIS/061/MOD16A2",
    #     "band": "PET",
    #     "cadence": "8D",
    #     "agg": "sum",
    #     "scale_m": 500,
    #     "description": "MODIS 8-day PET aggregated to monthly sum (×0.1 mm)."
    # },

    # Derived VPD from ERA5-Land MONTHLY_AGGR
    "era5_vpd": {
        "type": "derived_vpd",
        "source": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "cadence": "MONTHLY",
        "scale_m": 11132,
        "description": "Monthly VPD (kPa) from ERA5-Land 2m T and Td."
    },
}
