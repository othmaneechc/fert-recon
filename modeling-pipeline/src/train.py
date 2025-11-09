# src/train.py
import argparse, os, sys, json, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import HuberLoss, MSELoss
from tqdm import tqdm

from src.utils.io import load_config
from src.utils.feats import select_columns
from src.utils.metrics import rmse, mae, r2, metrics_extended
from src.utils.splits import temporal_split, temporal_leaveout, spatial_kfold
from src.data.sequence_dataset import MonthlySequenceDataset
from src.models.transformer import TransformerRegressor
from src.models.lstm import LSTMRegressor
from src.models.tcn import TCNRegressor
from src.models.gbdt import GBDTRegressor
from src.models.linear import LinearRegressor, ElasticNetRegressor
from src.data.tabular import aggregate_yearly

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from joblib import dump as joblib_dump

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

os.environ.setdefault("PYTHONUNBUFFERED", "1")


# ------------------------- logging to file + console ------------------------- #
class TeeStream:
    """Write to both the real stream and a logfile."""
    def __init__(self, real_stream, log_file):
        self.real = real_stream
        self.log = log_file
    def write(self, data):
        try:
            self.log.write(data)
        except Exception:
            pass
        self.real.write(data)
    def flush(self):
        try:
            self.log.flush()
        except Exception:
            pass
        self.real.flush()


# ------------------------- helpers ------------------------- #
def set_seed(seed: int):
    if seed is None:
        return
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _downsample_train_sequences(df_train, target_col, per_year_max, stratify="yield_decile", seed=1337):
    keys = df_train[["pixel_id", "year", target_col]].drop_duplicates()
    out_idx = []
    for yr, g in keys.groupby("year"):
        if len(g) <= per_year_max:
            out_idx.append(g[["pixel_id", "year"]])
            continue
        if stratify == "yield_decile":
            q = pd.qcut(g[target_col], q=10, labels=False, duplicates="drop")
            per_bin = max(per_year_max // (q.max() + 1), 1)
            sub = []
            for b in range(q.max() + 1):
                idx = g[q == b]
                take = min(len(idx), per_bin)
                sub.append(idx.sample(n=take, random_state=seed))
            sub = pd.concat(sub, ignore_index=True)
        else:
            sub = g.sample(n=per_year_max, random_state=seed)
        out_idx.append(sub[["pixel_id", "year"]])
    keep = pd.concat(out_idx, ignore_index=True).drop_duplicates()
    return df_train.merge(keep, on=["pixel_id", "year"], how="inner")


def _as_float(x, default=None):
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip().replace(",", ""))
        except ValueError:
            return default
    return default


def _apply_cleaning(cfg, df, dyn, yearly):
    c = cfg["data"]["cleaning"]
    tcol = cfg["data"]["target_col"]
    before = len(df)

    # drop NA target
    if c.get("drop_na_target", True):
        df = df[~df[tcol].isna()]

    # min_yield > 0 (robust cast)
    min_y = _as_float(c.get("min_yield", 0.0), 0.0)
    if min_y is not None and min_y > 0.0:
        df = df[df[tcol] > float(min_y)]

    # optional fertilizer presence
    if c.get("require_any_fertilizer", False):
        fert_cols = [col for col in yearly if col.startswith("fert_")]
        if fert_cols:
            df = df[(df[fert_cols].fillna(0.0).sum(axis=1) > 0)]

    print(f"[clean] basic filters: {before:,} → {len(df):,} (min_yield={min_y})")
    return df


def _train_tabular_model(cfg, df, dyn_cols, yearly_cols, static_cols, crop, run_dir):
    tcol = cfg["data"]["target_col"]
    id_cols = cfg["data"].get("id_cols", ["pixel_id", "year"])
    pre_cfg = cfg.get("preproc", {})
    target_cfg = cfg.get("target", {})

    extra_static = pre_cfg.get("static_cols", [])
    all_static = list(static_cols) + [c for c in extra_static if c not in static_cols]

    tab = aggregate_yearly(
        df,
        dyn_cols,
        yearly_cols,
        tcol,
        aggregate_to_year=pre_cfg.get("aggregate_to_year", "mean"),
        sum_features=pre_cfg.get("sum_features", []),
        sum_keywords=pre_cfg.get("sum_keywords", ["precip"]),
        include_prev_year_yield=pre_cfg.get("include_prev_year_yield", True),
        static_cols=all_static,
        seasonal_quartiles=pre_cfg.get("seasonal_quartiles", True),
        extra_stats=pre_cfg.get("extra_stats", []),
        lag_years=int(pre_cfg.get("lag_years", 0) or 0),
        add_fertilizer_interactions=pre_cfg.get("add_fertilizer_interactions", False),
        log1p_features=pre_cfg.get("log1p_features", []),
    )

    tab = tab[~tab[tcol].isna()].copy()
    if tab.empty:
        raise RuntimeError("Aggregated dataset is empty; check preprocessing settings.")

    split_cfg = cfg["data"]["split"]
    if split_cfg["type"] == "temporal":
        sp = split_cfg["temporal"]
        strategy = sp.get("strategy", "threshold")
        if strategy == "leaveout":
            tr, va, te = temporal_leaveout(tab, sp["val_year"], sp["test_year"])
            split_info = f"temporal-leaveout(val={sp['val_year']}, test={sp['test_year']})"
        else:
            tr, va, te = temporal_split(
                tab,
                sp["train_max_year"],
                sp["val_year"],
                sp["test_year_min"],
                test_year_max=sp.get("test_year_max"),
                train_min_year=sp.get("train_min_year"),
            )
            split_info = f"temporal(train<= {sp['train_max_year']}, val== {sp['val_year']}, test>={sp['test_year_min']})"
    else:
        tr, va, te = spatial_kfold(tab, **split_cfg["spatial"])
        split_info = "spatial-kfold"
    print(f"[split] {split_info}  | rows: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    qlo, qhi = cfg["data"]["cleaning"].get("yield_clip_quantiles", (None, None))
    if qlo is not None and qlo > 0:
        thr = tr[tcol].quantile(qlo)
        tr = tr[tr[tcol] >= thr]
        va = va[va[tcol] >= thr]
        te = te[te[tcol] >= thr]
    if qhi is not None and qhi < 1:
        thr = tr[tcol].quantile(qhi)
        tr = tr[tr[tcol] <= thr]
        va = va[va[tcol] <= thr]
        te = te[te[tcol] <= thr]

    print(f"[clean] tabular clip rows: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    exclude_cols = {tcol}
    exclude_cols.update(c for c in id_cols if c in tab.columns)
    feat_cols = [c for c in tab.columns if c not in exclude_cols]
    if not feat_cols:
        raise RuntimeError("No feature columns available for tabular model.")
    print(f"[features] tabular feature count={len(feat_cols)}")

    trX = tr[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    vaX = va[feat_cols].fillna(0.0).to_numpy(dtype=np.float32) if len(va) else np.zeros((0, len(feat_cols)), dtype=np.float32)
    teX = te[feat_cols].fillna(0.0).to_numpy(dtype=np.float32) if len(te) else np.zeros((0, len(feat_cols)), dtype=np.float32)

    log_transform = target_cfg.get("log_transform", False)
    log_offset = target_cfg.get("log_offset", 1.0 if log_transform else 0.0)

    def transform_y(arr):
        if not log_transform:
            return arr
        arr = np.asarray(arr, dtype=np.float32)
        safe = np.clip(arr + log_offset, 1e-6, None)
        return np.log(safe)

    def inverse_y(arr):
        if not log_transform:
            return np.asarray(arr)
        out = np.exp(arr) - log_offset
        return np.maximum(out, 0.0)

    y_tr = tr[tcol].to_numpy()
    y_va = va[tcol].to_numpy()
    y_te = te[tcol].to_numpy()

    y_tr_t = transform_y(y_tr)

    typ = cfg['model']['type']
    use_scaler = typ in {'linear', 'elasticnet'}

    if use_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(trX)
        X_va = scaler.transform(vaX) if len(vaX) else vaX
        X_te = scaler.transform(teX) if len(teX) else teX
    else:
        scaler = None
        X_tr, X_va, X_te = trX, vaX, teX

    if typ == 'linear':
        lin_cfg = cfg['model'].get('linear', {})
        alpha = lin_cfg.get('alpha', cfg['model'].get('alpha', 1.0))
        fit_intercept = lin_cfg.get('fit_intercept', True)
        model = LinearRegressor(alpha=alpha, fit_intercept=fit_intercept)
    elif typ == 'elasticnet':
        en_cfg = cfg['model'].get('elasticnet', {})
        model = ElasticNetRegressor(
            alpha=en_cfg.get('alpha', 1.0),
            l1_ratio=en_cfg.get('l1_ratio', 0.5),
            fit_intercept=en_cfg.get('fit_intercept', True),
            max_iter=en_cfg.get('max_iter', 2000),
            tol=en_cfg.get('tol', 1e-4),
        )
    elif typ == 'random_forest':
        rf_cfg = cfg['model'].get('random_forest', {})
        model = RandomForestRegressor(**rf_cfg)
    elif typ == 'gbrt':
        gbrt_cfg = cfg['model'].get('gbrt', {})
        model = GradientBoostingRegressor(**gbrt_cfg)
    elif typ == 'xgboost':
        if XGBRegressor is None:
            raise ImportError('xgboost is not installed; pip install xgboost to use this model.')
        xgb_cfg = cfg['model'].get('xgboost', {})
        model = XGBRegressor(**xgb_cfg)
    else:
        raise SystemExit(f'Unsupported tabular model type: {typ}')

    model.fit(X_tr, y_tr_t)

    def evaluate_split(X, y_true):
        if X.size == 0 or y_true.size == 0:
            return np.asarray([]), {}
        preds = model.predict(X)
        preds = inverse_y(preds)
        metrics = metrics_extended(y_true, preds)
        return preds, metrics

    _, train_metrics = evaluate_split(X_tr, y_tr)
    va_preds, val_metrics = evaluate_split(X_va, y_va)
    te_preds, test_metrics = evaluate_split(X_te, y_te)

    if val_metrics:
        print("[val] " + "  ".join([
            f"rmse={val_metrics['rmse']:.2f}",
            f"mae={val_metrics['mae']:.2f}",
            f"r2={val_metrics['r2']:.3f}",
            f"acc@20%={val_metrics['acc@20%']:.1f}%"
        ]))

    if test_metrics:
        print("[test] " + "  ".join([
            f"rmse={test_metrics['rmse']:.2f}",
            f"mae={test_metrics['mae']:.2f}",
            f"r2={test_metrics['r2']:.3f}",
            f"acc@20%={test_metrics['acc@20%']:.1f}%"
        ]))
    else:
        print("[test] no test samples")

    # per-year metrics on test
    per_year = {}
    if test_metrics and len(te):
        for yr, group in te.assign(pred=te_preds).groupby("year"):
            per_year[int(yr)] = metrics_extended(group[tcol].values, group["pred"].values)
        yr_rmse = sorted([(y, per_year[y]["rmse"]) for y in per_year], key=lambda t: t[1])
        print("[test] per-year RMSE (best→worst): " + ", ".join([f"{y}:{r:.1f}" for y, r in yr_rmse]))

    pred_id_cols = [c for c in id_cols if c in te.columns]
    preds_df = te[pred_id_cols + [tcol]].copy()
    if test_metrics and len(te_preds):
        preds_df["y_pred"] = te_preds
        preds_df["residual"] = te_preds - preds_df[tcol].to_numpy()
    pred_csv = run_dir / f"preds_{crop}_{cfg['model']['type']}_test.csv"
    preds_df.to_csv(pred_csv, index=False)

    out_json = {
        "overall": test_metrics,
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
        "per_year": per_year,
        "config_name": cfg.get("name", f"{crop}_{cfg['model']['type']}")
    }
    with open(run_dir / f"metrics_{crop}_{cfg['model']['type']}.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # Save scaler + model for reuse
    artefact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feat_cols,
        "log_transform": log_transform,
        "log_offset": log_offset,
    }
    joblib_dump(artefact, run_dir / f"model_{crop}_{cfg['model']['type']}.joblib")

    # Simple parity plot if we have data
    if test_metrics and len(te_preds):
        try:
            plt.figure(figsize=(5, 5))
            both = np.concatenate([y_te, te_preds])
            lo, hi = float(np.nanpercentile(both, 1)), float(np.nanpercentile(both, 99))
            lo, hi = min(lo, hi), max(lo, hi)
            plt.scatter(y_te, te_preds, s=5, alpha=0.4)
            plt.plot([lo, hi], [lo, hi], "k--", lw=1)
            plt.xlabel("True yield"); plt.ylabel("Predicted yield"); plt.title("Parity (test)")
            plt.tight_layout()
            plt.savefig(run_dir / f"parity_{crop}_{cfg['model']['type']}.png", dpi=160)
        except Exception as e:
            print(f"[plot] parity failed: {e}")

# ------------------------- training ------------------------- #
def train_one(cfg):
    # seeds
    set_seed(cfg.get("training", {}).get("seed", None))

    required_sections = ("data", "model", "training", "logging")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(
            "Config missing required sections: "
            + ", ".join(missing)
            + ". Did you point to a feature-set definition instead of a training config?"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build per-run directory and tee logs
    out_root = Path(cfg["logging"]["out_dir"])
    run_name = cfg.get("name") or f"{cfg['data']['crop']}_{cfg['model']['type']}"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.txt"
    with open(log_path, "w", buffering=1) as logf:
        # tee stdout/stderr; ensure tqdm writes to stdout
        sys.stdout = TeeStream(sys.__stdout__, logf)
        sys.stderr = TeeStream(sys.__stderr__, logf)

        t0 = time.time()

        crop = cfg["data"]["crop"]
        parquet_path = cfg["data"].get("parquet") or (cfg["data"]["wheat_path"] if crop == "wheat" else cfg["data"]["maize_path"])
        print("=" * 80)
        print(f"[setup] device={device}  crop={crop}  parquet={parquet_path}")
        print(f"[setup] run_dir={run_dir}")
        print("=" * 80)

        if not Path(parquet_path).exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        # Load
        df = pd.read_parquet(parquet_path)

        # Feature selection
        dyn_list = cfg["data"]["dynamic_features"]
        yearly_regex = cfg["data"]["yearly_regex"]
        static_regex = cfg["data"]["static_regex"]
        dyn, yearly, static = select_columns(df.columns, dyn_list, yearly_regex, static_regex)

        # CLEAN basic first
        df = _apply_cleaning(cfg, df, dyn, yearly)

        typ = cfg['model']['type']
        if typ in {'linear', 'elasticnet', 'random_forest', 'xgboost', 'gbrt'}:
            _train_tabular_model(cfg, df, dyn, yearly, static, crop, run_dir)
            print(f'[done] elapsed={time.time() - t0:.1f}s')
            print(f'[logs] log file → {log_path}')
            return

        # Split
        if cfg["data"]["split"]["type"] == "temporal":
            sp = cfg["data"]["split"]["temporal"]
            strategy = sp.get("strategy", "threshold")
            if strategy == "leaveout":
                tr, va, te = temporal_leaveout(df, sp["val_year"], sp["test_year"])
                split_info = f"temporal-leaveout(val={sp['val_year']}, test={sp['test_year']})"
            else:
                tr, va, te = temporal_split(
                    df,
                    sp["train_max_year"],
                    sp["val_year"],
                    sp["test_year_min"],
                    test_year_max=sp.get("test_year_max"),
                    train_min_year=sp.get("train_min_year"),
                )
                split_info = f"temporal(train<= {sp['train_max_year']}, val== {sp['val_year']}, test>={sp['test_year_min']})"
        else:
            tr, va, te = spatial_kfold(df, **cfg["data"]["split"]["spatial"])
            split_info = "spatial-kfold"
        print(f"[split] {split_info}  | sizes: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

        # TRAIN-ONLY yield clipping thresholds
        qlo, qhi = cfg["data"]["cleaning"]["yield_clip_quantiles"]
        tcol = cfg["data"]["target_col"]
        lo = tr[tcol].quantile(qlo) if qlo and qlo > 0 else None
        hi = tr[tcol].quantile(qhi) if qhi and qhi < 1 else None
        if lo is not None:
            tr = tr[tr[tcol] >= lo]; va = va[va[tcol] >= lo]; te = te[te[tcol] >= lo]
        if hi is not None:
            tr = tr[tr[tcol] <= hi]; va = va[va[tcol] <= hi]; te = te[te[tcol] <= hi]
        print(f"[clean] yield clip train-quantiles {qlo}-{qhi}: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

        # Optional downsample
        ds_cfg = cfg["data"]["cleaning"]["downsample"]
        if ds_cfg.get("enable", False):
            before = len(tr)
            tr = _downsample_train_sequences(
                tr, tcol,
                per_year_max=ds_cfg["per_year_max_sequences"],
                stratify=ds_cfg.get("stratify_by", "yield_decile"),
                seed=ds_cfg.get("seed", 1337),
            )
            print(f"[downsample] train rows: {before:,} → {len(tr):,}")

        # Build datasets with TRAIN-ONLY normalization
        min_dyn_months = cfg["data"]["cleaning"]["min_dynamic_months"]
        kw_common = dict(
            dynamic_cols=dyn,
            yearly_cols=yearly,
            static_cols=static,
            target_col=tcol,
            seq_len=cfg["data"]["seq_len"],
            min_months=cfg["data"]["min_months"],
            add_prev_year_yield=cfg["data"]["add_prev_year_yield"],
            normalization=cfg["data"]["normalization"]["scheme"],
            clip_std=cfg["data"]["normalization"]["clip_std"],
            history_years=cfg["data"].get("history_years", 1),
        )

        ds_tr = MonthlySequenceDataset(tr, **kw_common, precomputed_stats=None, fit_stats=True)
        if min_dyn_months and min_dyn_months > 0 and min_dyn_months != cfg['data']['min_months']:
            ds_tr = MonthlySequenceDataset(tr, **{**kw_common, 'min_months': min_dyn_months}, precomputed_stats=None, fit_stats=True)

        norm_stats = ds_tr.norm_stats

        va_years = set(va['year'].unique()) if len(va) else set()
        te_years = set(te['year'].unique()) if len(te) else set()
        va_context = pd.concat([tr, va], ignore_index=True) if len(va_years) else va.copy()
        te_context = pd.concat([tr, va, te], ignore_index=True) if len(te_years) else te.copy()

        ds_va = MonthlySequenceDataset(va_context, **kw_common, precomputed_stats=norm_stats, fit_stats=False, target_years=va_years if va_years else None)
        ds_te = MonthlySequenceDataset(te_context, **kw_common, precomputed_stats=norm_stats, fit_stats=False, target_years=te_years if te_years else None)

        print(f"[dataset] #samples: train={len(ds_tr):,}  val={len(ds_va):,}  test={len(ds_te):,}")
        if len(ds_tr) == 0:
            raise RuntimeError("Train dataset is EMPTY after cleaning/split. Check config and inputs.")

        static_dim = getattr(ds_tr, 'static_dim', len(static))
        in_dim = getattr(ds_tr, 'time_feature_dim', None) or (len(ds_tr.dynamic_cols) + (len(ds_tr.yearly_cols) if ds_tr.yearly_cols else 0))
        hist_years = cfg['data'].get('history_years', 1)
        print(f"[features] dynamic={len(ds_tr.dynamic_cols)} yearly={len(ds_tr.yearly_cols)} static={static_dim}  in_dim={in_dim}  history_years={hist_years}  seq_len={cfg['data']['seq_len']}")

        dl_tr = DataLoader(ds_tr, batch_size=cfg["data"]["batch_size"], shuffle=True,
                           num_workers=cfg["data"]["num_workers"], pin_memory=(device == "cuda"))
        dl_va = DataLoader(ds_va, batch_size=cfg["data"]["batch_size"], shuffle=False,
                           num_workers=cfg["data"]["num_workers"], pin_memory=(device == "cuda"))
        dl_te = DataLoader(ds_te, batch_size=cfg["data"]["batch_size"], shuffle=False,
                           num_workers=cfg["data"]["num_workers"], pin_memory=(device == "cuda"))

        # Model selection
        typ = cfg["model"]["type"]
        if typ == "transformer":
            mcfg = cfg["model"]["transformer"]
            model = TransformerRegressor(
                in_dim,
                d_model=mcfg["d_model"],
                nhead=mcfg["nhead"],
                n_layers=mcfg["n_layers"],
                dim_ff=mcfg["dim_feedforward"],
                dropout=mcfg["dropout"],
                posenc=mcfg["posenc"],
                head_hidden=cfg["model"]["head_hidden"],
                head_layers=cfg["model"]["head_layers"],
                static_dim=static_dim if mcfg.get("use_static_token", True) else 0,
                use_film=mcfg.get("use_film", True)
            ).to(device)
        elif typ == "lstm":
            mcfg = cfg["model"]["lstm"]
            model = LSTMRegressor(
                in_dim,
                hidden_size=mcfg["hidden_size"],
                num_layers=mcfg["num_layers"],
                dropout=mcfg["dropout"],
                bidirectional=mcfg["bidirectional"],
                head_hidden=cfg["model"]["head_hidden"],
                head_layers=cfg["model"]["head_layers"],
            ).to(device)
        elif typ == "tcn":
            mcfg = cfg["model"]["tcn"]
            model = TCNRegressor(
                in_dim,
                channels=mcfg["channels"],
                kernel_size=mcfg["kernel_size"],
                dropout=mcfg["dropout"],
                head_hidden=cfg["model"]["head_hidden"],
                head_layers=cfg["model"]["head_layers"],
            ).to(device)
        elif typ == "gbdt":
            print("[warn] GBDT baseline bypasses sequence cleaning; run with gbdt_base.yaml when needed.")
            return
        else:
            raise SystemExit("Unknown model type.")

        # Loss/opt
        loss_name = cfg["training"]["loss"]
        criterion = MSELoss() if loss_name == "mse" else HuberLoss(delta=1.0)
        optim = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

        # Train
        best_rmse, patience, epochs_no_improve = 1e9, cfg["training"]["patience"], 0
        E = cfg["training"]["epochs"]
        for epoch in range(E):
            model.train()
            run_loss, nobs = 0.0, 0
            pbar = tqdm(dl_tr, desc=f"epoch {epoch+1}/{E} [train]", leave=False, file=sys.stdout)
            for batch in pbar:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                m = batch["month_idx"].to(device)
                s = batch.get("static")
                if s is not None:
                    s = s.to(device)
                pred = model(x, month_idx=m, static=s)
                if pred.shape != y.shape:
                    y = y.view_as(pred)
                loss = criterion(pred, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                run_loss += float(loss.detach().cpu().item()) * x.size(0)
                nobs += x.size(0)
                pbar.set_postfix(loss=run_loss / max(1, nobs))

            # Validation
            model.eval()
            Y, P = [], []
            with torch.no_grad():
                for batch in tqdm(dl_va, desc=f"epoch {epoch+1}/{E} [val]", leave=False, file=sys.stdout):
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    m = batch["month_idx"].to(device)
                    s = batch.get("static")
                    if s is not None:
                        s = s.to(device)
                    p = model(x, month_idx=m, static=s)
                    Y.append(y.cpu().numpy()); P.append(p.cpu().numpy())
            if len(Y) == 0:
                cur_rmse = float("inf")
                print(f"[val] epoch {epoch+1}/{E}  (no validation samples)")
            else:
                Y = np.concatenate(Y).reshape(-1)
                P = np.concatenate(P).reshape(-1)
                mvals = metrics_extended(Y, P)
                cur_rmse = mvals["rmse"]
                msg = "  ".join([
                    f"rmse={mvals['rmse']:.2f}",
                    f"mae={mvals['mae']:.2f}",
                    f"r2={mvals['r2']:.3f}",
                    f"mape%={mvals['mape%']:.1f}",
                    f"r={mvals['pearson_r']:.3f}",
                    f"acc@20%={mvals['acc@20%']:.1f}%"
                ])
                print(f"[val] epoch {epoch+1}/{E}  {msg}")

            # Early stopping
            if cur_rmse + 1e-6 < best_rmse:
                best_rmse = cur_rmse
                epochs_no_improve = 0
                torch.save(model.state_dict(), str(run_dir / f"best_{crop}_{typ}.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[early-stop] no improvement for {patience} epochs (best_rmse={best_rmse:.4f})")
                    break

        # ---------- Test ----------
        ckpt_path = run_dir / f"best_{crop}_{typ}.pt"
        if not ckpt_path.exists():
            torch.save(model.state_dict(), str(ckpt_path))
        try:
            state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state)
        model.eval()

        Yt, Pt, PIDs, YEARS = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dl_te, desc="[test]", leave=False, file=sys.stdout):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                m = batch["month_idx"].to(device)
                s = batch.get("static")
                if s is not None:
                    s = s.to(device)
                p = model(x, month_idx=m, static=s)
                Yt.append(y.cpu().numpy()); Pt.append(p.cpu().numpy())
                if "pixel_id" in batch: PIDs.append(batch["pixel_id"].cpu().numpy())
                if "year" in batch:     YEARS.append(batch["year"].cpu().numpy())

        Yt = np.concatenate(Yt).reshape(-1) if len(Yt) else np.asarray([])
        Pt = np.concatenate(Pt).reshape(-1) if len(Pt) else np.asarray([])
        metrics = metrics_extended(Yt, Pt) if Yt.size else {}

        if metrics:
            print("[test] " + "  ".join([
                f"rmse={metrics['rmse']:.2f}",
                f"mae={metrics['mae']:.2f}",
                f"r2={metrics['r2']:.3f}",
                f"mape%={metrics['mape%']:.1f}",
                f"r={metrics['pearson_r']:.3f}",
                f"acc@20%={metrics['acc@20%']:.1f}%"
            ]))
        else:
            print("[test] no test samples")

        # per-year breakdown
        per_year = {}
        if YEARS:
            years_vec = np.concatenate(YEARS).reshape(-1)
            for yr in sorted(np.unique(years_vec)):
                idx = (years_vec == yr)
                per_year[int(yr)] = metrics_extended(Yt[idx], Pt[idx])

            # print a quick summary: top/bottom by RMSE
            yr_rmse = sorted([(y, per_year[y]["rmse"]) for y in per_year], key=lambda t: t[1])
            print("[test] per-year RMSE (best→worst): " + ", ".join([f"{y}:{r:.1f}" for y, r in yr_rmse]))

        # save predictions CSV
        pred_rows = {"y_true": Yt, "y_pred": Pt, "residual": (Pt - Yt)}
        if PIDs:  pred_rows["pixel_id"] = np.concatenate(PIDs).reshape(-1)
        if YEARS: pred_rows["year"] = np.concatenate(YEARS).reshape(-1)
        pred_df = pd.DataFrame(pred_rows)
        pred_csv = run_dir / f"preds_{crop}_{typ}_test.csv"
        pred_df.to_csv(pred_csv, index=False)

        # save metrics JSON
        out_json = {
            "overall": metrics,
            "per_year": per_year,
            "config_name": cfg.get("name", f"{crop}_{typ}"),
            "crop": crop,
            "model": typ,
            "data_info": {
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "test_rows": int(len(te)),
                "train_samples": int(len(ds_tr)),
                "val_samples": int(len(ds_va)),
                "test_samples": int(len(ds_te)),
            }
        }
        with open(run_dir / f"metrics_{crop}_{typ}.json", "w") as f:
            json.dump(out_json, f, indent=2)

        # parity plot
        try:
            if Yt.size and Pt.size:
                plt.figure(figsize=(5, 5))
                both = np.concatenate([Yt, Pt])
                lo, hi = float(np.nanpercentile(both, 1)), float(np.nanpercentile(both, 99))
                lo, hi = min(lo, hi), max(lo, hi)
                plt.scatter(Yt, Pt, s=3, alpha=0.3)
                plt.plot([lo, hi], [lo, hi], "k--", lw=1)
                plt.xlabel("True yield"); plt.ylabel("Predicted yield"); plt.title("Parity (test)")
                plt.tight_layout()
                plt.savefig(run_dir / f"parity_{crop}_{typ}.png", dpi=160)
        except Exception as e:
            print(f"[plot] parity failed: {e}")

        # residuals by year
        try:
            if YEARS and Yt.size:
                years_vec = np.concatenate(YEARS).reshape(-1)
                df_tmp = pd.DataFrame({"year": years_vec, "res": Pt - Yt})
                g = df_tmp.groupby("year")["res"].agg(["mean", "median", "std", "count"]).reset_index()
                g.to_csv(run_dir / f"residuals_by_year_{crop}_{typ}.csv", index=False)
                plt.figure(figsize=(6, 3))
                plt.plot(g["year"], g["mean"], marker="o")
                plt.axhline(0, color="k", lw=1, ls="--")
                plt.xlabel("Year"); plt.ylabel("Mean residual"); plt.title("Residuals by year (test)")
                plt.tight_layout()
                plt.savefig(run_dir / f"residuals_by_year_{crop}_{typ}.png", dpi=160)
        except Exception as e:
            print(f"[plot] residuals_by_year failed: {e}")

        ckpt_path = run_dir / f"best_{crop}_{typ}.pt"
        print(f"[done] elapsed={time.time() - t0:.1f}s  saved={ckpt_path}")
        print(f"[logs] log file → {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file (supports include + ${...})")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_one(cfg)


if __name__ == "__main__":
    main()
