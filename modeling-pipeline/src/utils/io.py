# src/utils/io.py
import os
from pathlib import Path
from typing import Any, Dict, Union, List
import re
import yaml

_VAR_RE = re.compile(r"\$\{([^}]+)\}")

def _read_yaml(p: Union[str, Path]) -> Dict[str, Any]:
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    return data or {}

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _merge_includes(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    inc = cfg.pop("include", []) or []
    if isinstance(inc, str):
        inc = [inc]
    merged: Dict[str, Any] = {}
    # load each include (supports recursive include inside included files)
    for rel in inc:
        inc_path = (base_dir / rel).resolve()
        inc_cfg = _read_yaml(inc_path)
        inc_cfg = _merge_includes(inc_cfg, inc_path.parent)
        _deep_update(merged, inc_cfg)
    # then override with this file's keys
    _deep_update(merged, cfg)
    return merged

def _lookup_ctx(path: str, cfg: Dict[str, Any], fallback: Any = None) -> Any:
    """
    Look up dotted path like 'data.crop' in cfg. If not dotted, try top-level then env.
    """
    cur: Any = cfg
    if "." in path:
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return fallback
        return cur
    # plain key: try cfg['data'][key] then top-level, else env
    if "data" in cfg and isinstance(cfg["data"], dict) and path in cfg["data"]:
        return cfg["data"][path]
    if path in cfg:
        return cfg[path]
    return os.environ.get(path, fallback)

def _interpolate(obj: Any, cfg: Dict[str, Any]) -> Any:
    """Recursively replace ${var} or ${data.var} in strings using cfg/env."""
    if isinstance(obj, dict):
        return {k: _interpolate(v, cfg) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, cfg) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            key = m.group(1).strip()
            val = _lookup_ctx(key, cfg, fallback=m.group(0))  # leave token if not found
            return str(val)
        return _VAR_RE.sub(repl, obj)
    return obj

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a YAML config with:
      - support for `include:` (relative to the file)
      - deep merge (later files override earlier)
      - ${...} interpolation (e.g., yield_${crop} or ${data.crop})
    """
    path = Path(path).resolve()
    cfg0 = _read_yaml(path)
    merged = _merge_includes(cfg0, path.parent)
    # final interpolation pass (after include + merge)
    merged = _interpolate(merged, merged)
    return merged
