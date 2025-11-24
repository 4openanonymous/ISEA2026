#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W4P — Param Explorer (strict)
- Purpose: from W2 events, recompute a model-driven time column (dt_model) under
           externally specified (gamma, epsilon, lambda) WITHOUT touching W2/W3.
- Input : --csv  path to events.csv (W2). A W3 merged CSV is tolerated but only
           required columns are used.
- Output: --out  CSV that preserves all original columns + adds <dt_out> (default: dt_model)
          plus a JSON sidecar with provenance & parameters for audit.

Required columns (with aliases):
  run_id  : ["run_id","runID","run"]
  id      : ["id","event_id"]
  d_hat   : ["d_hat","dhat","d̂"]
  rho     : ["rho","ρ"]
  t_min   : ["t_min","tmin"]
  t_max   : ["t_max","tmax"]
Optional:
  order   : ["order_idx","order","idx"]  # used to compute EMA sequence

Model:
  base_i = t_min + (t_max - t_min) * (d_hat_i ** gamma)
  ema_i  = EMA(base, epsilon)           # EMA(new) = (1-eps)*curr + eps*prev
  dt_i   = ema_i * (1 + lambda * rho_i)

Usage:
  python -m scripts.w4_param_explorer \
    --csv reports/W2_runs/2021/events.csv \
    --out reports/W4_variants/2021/g1.2_e0.3_l0.7/events_dt_model.csv \
    --gamma 1.2 --epsilon 0.3 --lambda 0.7

  # custom output column & order column
  python -m scripts.w4_param_explorer \
    --csv path/to/events.csv --out path/to/out.csv \
    --gamma 1.0 --epsilon 0.2 --lambda 0.6 \
    --dt-out dt_model_g1.0e0.2l0.6 --order-col order_idx
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd

ALIAS = {
    "run_id": ["run_id","runID","run"],
    "id":     ["id","event_id"],
    "d_hat":  ["d_hat","dhat","d̂"],
    "rho":    ["rho","ρ"],
    "t_min":  ["t_min","tmin"],
    "t_max":  ["t_max","tmax"],
    "order":  ["order_idx","order","idx"],
}

def _pick(df: pd.DataFrame, key: str) -> str:
    cols = {c.strip(): c for c in df.columns}
    candidates = ALIAS[key]
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    # try case-insensitive fallback
    lowmap = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowmap:
            return lowmap[cand.lower()]
    raise KeyError(f"Missing required column for '{key}'. Tried aliases: {candidates}. Have: {list(df.columns)}")

def _to_float_series(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().any():
        bad = int(s.isna().sum())
        raise ValueError(f"Column '{col}' → numeric failed for {bad} rows.")
    return s

def _ema(arr: np.ndarray, eps: float) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    prev = float(arr[0])
    out[0] = prev
    for i in range(1, len(arr)):
        curr = float(arr[i])
        prev = (1.0 - eps) * curr + eps * prev
        out[i] = prev
    return out

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input W2 events.csv (or W3 merged; extra cols ignored)")
    ap.add_argument("--out", required=True, help="output CSV (will create parent dirs)")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.6)
    ap.add_argument("--dt-out", default="dt_model", help="name of output time column")
    ap.add_argument("--order-col", default=None, help="override order column; else try aliases or fall back to original row order")
    args = ap.parse_args()

    src = Path(args.csv)
    if not src.exists():
        raise FileNotFoundError(f"Input CSV not found: {src}")
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)

    # Resolve columns
    col_run  = _pick(df, "run_id")
    col_id   = _pick(df, "id")
    col_d    = _pick(df, "d_hat")
    col_rho  = _pick(df, "rho")
    col_tn   = _pick(df, "t_min")
    col_tx   = _pick(df, "t_max")
    if args.order_col is not None:
        if args.order_col not in df.columns:
            raise KeyError(f"--order-col '{args.order_col}' not in columns: {list(df.columns)}")
        col_ord = args.order_col
    else:
        # optional
        try:
            col_ord = _pick(df, "order")
        except KeyError:
            col_ord = None

    # Strong typing
    d_hat = _to_float_series(df, col_d, "d_hat")
    rho   = _to_float_series(df, col_rho, "rho")
    t_min = _to_float_series(df, col_tn, "t_min")
    t_max = _to_float_series(df, col_tx, "t_max")

    # 1) base
    gamma = float(args.gamma)
    eps   = float(args.epsilon)
    lam   = float(args.lam)
    base = t_min + (t_max - t_min) * np.power(d_hat.astype(float), gamma)

    # 2) EMA by order (if any), else by original order
    if col_ord:
        order_index = df[col_ord].astype(float).argsort(kind="mergesort")
        base_sorted = base.iloc[order_index].to_numpy()
        ema_sorted  = _ema(base_sorted, eps)
        ema_series  = pd.Series(ema_sorted, index=base.index[order_index]).sort_index()
        ema_seq = ema_series
    else:
        ema_seq = pd.Series(_ema(base.to_numpy(), eps), index=base.index)

    # 3) boundary friction
    dt_model = ema_seq * (1.0 + lam * rho.astype(float))
    dt_model.name = args.dt_out

    # Output CSV (preserve all columns, append dt_model)
    df_out = df.copy()
    if args.dt_out in df_out.columns:
        raise ValueError(f"Output column '{args.dt_out}' already exists in input. Choose another via --dt-out.")
    df_out[args.dt_out] = dt_model
    df_out.to_csv(outp, index=False, encoding="utf-8")
    print(f"[W4P] wrote {outp}")

    # Sidecar JSON for audit
    meta = {
        "source_csv": str(src),
        "source_sha256": _sha256_file(src),
        "rows": int(len(df_out)),
        "required_columns": {
            "run_id": col_run, "id": col_id, "d_hat": col_d, "rho": col_rho, "t_min": col_tn, "t_max": col_tx
        },
        "order_column": col_ord,
        "params": {"gamma": gamma, "epsilon": eps, "lambda": lam},
        "dt_out": args.dt_out,
        "notes": [
            "Only required columns used for computation; extra columns preserved verbatim.",
            "EMA(new)=(1-epsilon)*current + epsilon*previous"
        ],
    }
    with (outp.with_suffix(outp.suffix + ".json")).open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[W4P] wrote {outp.with_suffix(outp.suffix + '.json')}")

if __name__ == "__main__":
    main()