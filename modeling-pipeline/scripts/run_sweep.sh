#!/usr/bin/env bash
set -euo pipefail

CONF="$1"   # e.g., configs/ablations/arch_vs_seq.yaml
PY=${PYTHON:-python}

# Simple sweep runner: expands the grid and calls train.py for each setting
$PY - "$CONF" <<'PY'
import os, sys, json, itertools, copy, re
from pathlib import Path
import yaml, subprocess, tempfile, copy as _copy

def load_yaml(path):
    with open(path) as f: return yaml.safe_load(f)

def deep_merge(base, override):
    if base is None:
        base = {}
    result = _copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = _copy.deepcopy(v)
    return result

if len(sys.argv) < 2:
    raise SystemExit("Usage: run_sweep.sh <config.yaml>")

conf_path = Path(sys.argv[1]).resolve()
conf = load_yaml(conf_path)

def flatten(d, prefix=""):
    out=[]
    for k,v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out += flatten(v, key)
        else:
            out.append((key, v))
    return out

def set_by_keypath(d, keypath, value):
    parts = keypath.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value

base_cfgs = []
conf_dir = conf_path.parent

for p in conf["sweep"]["base_configs"]:
    p_path = (conf_dir / p).resolve() if not os.path.isabs(p) else Path(p)
    b = load_yaml(p_path)
    # apply includes
    if "include" in b:
        merged = {}
        for inc in b["include"]:
            inc_path = (p_path.parent / inc).resolve() if not os.path.isabs(inc) else Path(inc)
            merged = deep_merge(merged, load_yaml(inc_path))
        b_wo_inc = {k:v for k,v in b.items() if k != "include"}
        b = deep_merge(merged, b_wo_inc)
    base_cfgs.append(b)

grid_items = list(conf["sweep"].get("grid", {}).items())
keys = [k for k,_ in grid_items]
vals = [v for _,v in grid_items]
combos = list(itertools.product(*vals)) if keys else [()]

case_defs = []
for case in conf["sweep"].get("cases", []):
    if isinstance(case, dict):
        label = case.get("name")
        overrides = case.get("overrides", {k:v for k,v in case.items() if k != "name"})
    else:
        label, overrides = None, case
    case_defs.append((label, overrides))
if not case_defs:
    case_defs = [(None, {})]

jobs=[]
for base in base_cfgs:
    for combo in combos:
        cur = copy.deepcopy(base)
        for k,val in zip(keys, combo):
            set_by_keypath(cur, k, val)
        for label, overrides in case_defs:
            job = copy.deepcopy(cur)
            for k,val in flatten(overrides):
                set_by_keypath(job, k, val)
            if label and "name" not in overrides:
                set_by_keypath(job, "name", label)
            jobs.append(job)

print(f"Running {len(jobs)} jobs...", flush=True)
for i,job in enumerate(jobs,1):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(job, tf)
        tf.flush()
        print(f"[{i}/{len(jobs)}] {tf.name}", flush=True)
        subprocess.run([sys.executable, "-m", "src.train", "--config", tf.name], check=True)
        os.unlink(tf.name)
PY
