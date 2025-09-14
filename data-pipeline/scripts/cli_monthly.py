#!/usr/bin/env python3
"""
Monthly CLI for Time Series Data Collection
==========================================

CLI that exports monthly aggregations for dynamic features.
This supports the transformer-based modeling pipeline.
"""

import argparse
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import modules lazily to avoid hanging on --help
def lazy_import():
    from exporters.soilgrids_exporter import run_soilgrids
    from exporters.region_monthly_exporter import run_region_monthly
    from exporters.point_exporter import run_point
    return run_soilgrids, run_region_monthly, run_point

from config.config import DEFAULT_COUNTRY, DEFAULT_START_YEAR, DEFAULT_END_YEAR, DEFAULT_OUTPUT_ROOT, MAX_WORKERS

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Monthly Time Series Pipeline CLI')
    sub = p.add_subparsers(dest='command', required=True)

    soil = sub.add_parser('soilgrids')
    soil.add_argument('--country', default=DEFAULT_COUNTRY)
    soil.add_argument('--out', default=f"{DEFAULT_OUTPUT_ROOT}/soilgrids")

    region = sub.add_parser('region')
    region.add_argument('dataset')
    region.add_argument('--start-date', required=True, help='Start date YYYY-MM-DD')
    region.add_argument('--end-date', required=True, help='End date YYYY-MM-DD')
    region.add_argument('--out', default=f"{DEFAULT_OUTPUT_ROOT}/region")
    region.add_argument('--country', default=DEFAULT_COUNTRY)
    region.add_argument('--aggregation', default='mean', choices=['mean', 'median', 'sum', 'max', 'min'],
                       help='Temporal aggregation method for monthly composites')

    point = sub.add_parser('point')
    point.add_argument('dataset')
    point.add_argument('--coords', default='coords.csv')
    point.add_argument('--out', default=f"{DEFAULT_OUTPUT_ROOT}/point")
    point.add_argument('--start-date', required=True)
    point.add_argument('--end-date', required=True)
    point.add_argument('--parallel', action='store_true')
    point.add_argument('--workers', type=int, default=MAX_WORKERS)

    args = p.parse_args()
    
    # Import the functions only when needed
    run_soilgrids, run_region_monthly, run_point = lazy_import()
    
    if args.command == 'soilgrids':
        run_soilgrids(args.out, args.country)
    elif args.command == 'region':
        run_region_monthly(args.dataset, args.start_date, args.end_date, 
                         args.out, args.country, args.aggregation)
    elif args.command == 'point':
        run_point(args.coords, args.dataset, args.start_date, args.end_date, 
                 512, 512, 'RGB', False, args.out, args.parallel, args.workers, False)