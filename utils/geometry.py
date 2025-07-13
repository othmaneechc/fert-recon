# Geometry helpers for bounding boxes & country AOI
import geopandas as gpd
from config import NATURAL_EARTH_GEOJSON
import math

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
