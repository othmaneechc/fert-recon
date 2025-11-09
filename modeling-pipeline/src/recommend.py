import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.utils.io import load_config
from src.utils.feats import select_columns
from src.data.sequence_dataset import MonthlySequenceDataset
from src.models.transformer import TransformerRegressor

def recommend(cfg_path, model_ckpt, out_csv, bounds="p01_p99"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(cfg_path)
    crop = cfg["data"]["crop"]
    p = cfg["data"]["wheat_path"] if crop=="wheat" else cfg["data"]["maize_path"]
    df = pd.read_parquet(p)

    dyn, yearly, static = select_columns(
        df.columns,
        cfg["data"]["dynamic_features"],
        cfg["data"]["yearly_regex"],
        cfg["data"].get("static_regex", "")
    )
    # Build dataset for the latest test year
    te_year = cfg["data"]["split"]["temporal"]["test_year_min"]
    history_years = cfg['data'].get('history_years', 1)
years_needed = list(range(te_year - history_years + 1, te_year + 1))
sub = df[df.year.isin(years_needed)].copy()

    ds = MonthlySequenceDataset(
        sub, dyn, yearly, cfg['data']['target_col'],
        seq_len=cfg['data']['seq_len'], min_months=cfg['data']['min_months'],
        add_prev_year_yield=cfg['data']['add_prev_year_yield'],
        normalization=cfg['data']['normalization']['scheme'],
        clip_std=cfg['data']['normalization']['clip_std'],
        static_cols=static,
        history_years=history_years
    )
    in_dim = getattr(ds, 'time_feature_dim', len(ds.dynamic_cols) + (len(ds.yearly_cols) if ds.yearly_cols else 0))
    static_dim = getattr(ds, 'static_dim', len(static))

    # Fertilizer column indices inside the model input
    fert_cols = [c for c in ds.yearly_cols if c.startswith("fert_")]
    base_offset = len(ds.dynamic_cols)
    fert_positions = {c: base_offset + ds.yearly_cols.index(c) for c in fert_cols}

    # Bounds from the raw dataframe (before normalization)
    lo, hi = {}, {}
    for c in fert_cols:
        lo[c] = float(df[c].quantile(0.01)); hi[c] = float(df[c].quantile(0.99))

    # Model
    mcfg = cfg["model"]["transformer"]
    model = TransformerRegressor(
        in_dim,
        d_model=mcfg["d_model"], nhead=mcfg["nhead"],
        n_layers=mcfg["n_layers"], dim_ff=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"], posenc=mcfg["posenc"],
        head_hidden=cfg["model"]["head_hidden"], head_layers=cfg["model"]["head_layers"],
        static_dim=static_dim if mcfg.get("use_static_token", True) else 0,
        use_film=mcfg.get("use_film", True)
    ).to(device)
    state = torch.load(model_ckpt, map_location=device); model.load_state_dict(state)
    model.eval()

    rows=[]
    it = tqdm(range(len(ds)), desc="[recommend]")
    for i in it:
        item = ds[i]
        year_val = int(item['year'].item()) if hasattr(item['year'], 'item') else int(item['year'])
        if year_val != te_year:
            continue
        x = item["x"].to(device)          # [T, F]
        m = item["month_idx"].to(device)  # [T]
        s = item.get("static")
        s = s.to(device) if s is not None and s.numel() > 0 else None
        base_pred = model(x.unsqueeze(0), month_idx=m.unsqueeze(0), static=None if s is None else s.unsqueeze(0)).item()

        # coordinate ascent over fertilizer features
        best = base_pred
        best_vals = {}
        cur = x.clone()
        for _ in range(3):  # 3 passes
            for c in fert_cols:
                jidx = fert_positions[c]
                # naive grid
                trial_vals = np.linspace(lo[c], hi[c], 7, dtype=np.float32)
                cand_best, cand_val = best, None
                for v in trial_vals:
                    cur[:, jidx] = torch.tensor(v, device=device)
                    p = model(cur.unsqueeze(0), month_idx=m.unsqueeze(0), static=None if s is None else s.unsqueeze(0)).item()
                    if p > cand_best:
                        cand_best, cand_val = p, v
                if cand_val is not None:
                    best, best_vals[c] = cand_best, cand_val
                    x = cur.clone()  # accept improvement

        rows.append({
            "pixel_id": int(item["pixel_id"].item()) if hasattr(item["pixel_id"], 'item') else int(item["pixel_id"]),
            "year": year_val,
            "pred_base": base_pred,
            "pred_opt": best,
            **{k: best_vals.get(k, np.nan) for k in fert_cols}
        })

    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[done] wrote recommendations → {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    recommend(args.config, args.ckpt, args.out_csv)
