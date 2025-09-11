# Geometry helpers for bounding boxes & country AOI
import geopandas as gpd
import math
import sys
import os

# Add the data-pipeline config to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data-pipeline'))
from config.config import NATURAL_EARTH_GEOJSON

def country_bbox(country_name: str, crs: str = "EPSG:4326") -> list:
    """Return [minx, miny, maxx, maxy] for given country."""
    fc = gpd.read_file(NATURAL_EARTH_GEOJSON)
    country = fc[fc.ADMIN == country_name].to_crs(crs)
    return list(country.total_bounds)


def point_bbox(lat: float, lon: float, pixels: int, resolution: float) -> tuple:
    """Return (lon_min, lon_max, lat_min, lat_max) around point."""
    earth_radius = 6371000
    ang = math.degrees(0.5 * ((pixels * resolution) / earth_radius))
    return (lon - ang, lon + ang, lat - ang, lat + ang)
