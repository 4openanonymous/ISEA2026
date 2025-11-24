#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

def to_f(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except Exception:
        return None

def norm_pm(x):
    if pd.isna(x): return ""
    s = str(x).strip().lower()
    if s in ("proxy","measured"): return s
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to summary_w2.csv")
    ap.add_argument("--rc-threshold", type=float, default=2.0, help="two-step 門檻（用於參考，不改寫）")
    ap.add_argument("--inplace", action="store_true", help="就地覆寫原 CSV（預設輸出 *_sanitized.csv）")
    args = ap.parse_args()

    p = Path(args.csv)
    df = pd.read_csv(p)

    # 1) FI/CV：若 min > max → 交換
    def fix_interval(lo, hi):
        a, b = to_f(lo), to_f(hi)
        if a is not None and b is not None and a > b:
            return b, a
        return lo, hi

    df["fi_min"], df["fi_max"] = zip(*[fix_interval(a, b) for a, b in zip(df["fi_min"], df["fi_max"])])
    df["cv_min"], df["cv_max"] = zip(*[fix_interval(a, b) for a, b in zip(df["cv_min"], df["cv_max"])])

    # 2) proxy_measured 正規化
    df["proxy_measured"] = [norm_pm(x) for x in df["proxy_measured"]]

    # 3) two-step 未通過 → 清空 λ* 與 band
    def clean_lambda(row):
        rc_two = str(row.get("rc_two_step","")).strip().lower()
        passed = rc_two in ("true","t","1","yes","y","pass")
        if not passed:
            row["lam_star"] = ""
            row["lam_star_band_low"] = ""
            row["lam_star_band_high"] = ""
        return row

    df = df.apply(clean_lambda, axis=1)

    # 4) 產出
    if args.inplace:
        out_csv = p
    else:
        out_csv = p.with_name(p.stem + "_sanitized.csv")

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[sanitize] wrote: {out_csv}")

if __name__ == "__main__":
    main()