#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List

def collect_metrics(paths: List[Path], metric: str) -> List[float]:
    values = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
            overall = data.get("overall", {})
            if metric in overall:
                values.append(float(overall[metric]))
        except Exception as exc:
            print(f"[warn] failed to parse {p}: {exc}")
    return values

def main():
    ap = argparse.ArgumentParser(description="Aggregate metric across sweep outputs")
    ap.add_argument("log_dir", type=Path, help="Directory containing run subfolders with metrics_*.json")
    ap.add_argument("--metric", default="acc@20%", help="Metric key inside the metrics JSON (default: acc@20%)")
    args = ap.parse_args()

    metric_files = sorted(args.log_dir.glob("**/metrics_*.json"))
    if not metric_files:
        raise SystemExit(f"No metrics_*.json files found under {args.log_dir}")

    values = collect_metrics(metric_files, args.metric)
    if not values:
        raise SystemExit(f"Metric '{args.metric}' not found in any metrics file under {args.log_dir}")

    mean_val = sum(values) / len(values)
    print(f"metric={args.metric}  n={len(values)}  mean={mean_val:.4f}")

if __name__ == "__main__":
    main()
