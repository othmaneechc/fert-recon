# Command-line interface
import argparse
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import modules lazily to avoid hanging on --help
def lazy_import():
    from exporters.soilgrids_exporter import run_soilgrids
    from exporters.region_exporter import run_region
    from exporters.point_exporter import run_point
    return run_soilgrids, run_region, run_point

from config.config import DEFAULT_COUNTRY, DEFAULT_START_YEAR, DEFAULT_END_YEAR, DEFAULT_OUTPUT_ROOT, MAX_WORKERS

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Fertilizer pipeline CLI')
    sub = p.add_subparsers(dest='command', required=True)

    soil = sub.add_parser('soilgrids')
    soil.add_argument('--country', default=DEFAULT_COUNTRY)
    soil.add_argument('--out', default=f"{DEFAULT_OUTPUT_ROOT}/soilgrids")

    region = sub.add_parser('region')
    region.add_argument('dataset')
    region.add_argument('--year', type=int, default=DEFAULT_START_YEAR)
    region.add_argument('--out', default=f"{DEFAULT_OUTPUT_ROOT}/region")
    region.add_argument('--country', default=DEFAULT_COUNTRY)

    point = sub.add_parser('point')
    point.add_argument('dataset')
    point.add_argument('--coords', default='coords.csv')
    point.add_argument('--out', default=f"{DEFAULT_OUTPUT_ROOT}/point")
    point.add_argument('--year', type=int, default=DEFAULT_START_YEAR)
    point.add_argument('--parallel', action='store_true')
    point.add_argument('--workers', type=int, default=MAX_WORKERS)

    args = p.parse_args()
    
    # Import the functions only when needed
    run_soilgrids, run_region, run_point = lazy_import()
    
    if args.command == 'soilgrids':
        run_soilgrids(args.out, args.country)
    elif args.command == 'region':
        start = f"{args.year}-01-01"; end = f"{args.year}-12-31"
        run_region(args.dataset, start, end, args.out, args.country)
    elif args.command == 'point':
        start = f"{args.year}-01-01"; end = f"{args.year}-12-31"
        run_point(args.coords, args.dataset, start, end, 512, 512, 'RGB', False, args.out, args.parallel, args.workers, False)
