#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys, math
from pathlib import Path
import pandas as pd
import numpy as np

REQ_COLS = [
    "year","fi_min","fi_max","cv_min","cv_max","eps_star",
    "lam_star_band_low","lam_star_band_high","lam_star",
    "rc_max","rc_at","rc_two_step","proxy_measured",
    "rc_series","path"
]

def to_f(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except Exception:
        return None

def to_b(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x in ("true","t","1","yes","y","pass"): return True
        if x in ("false","f","0","no","n","fail"): return False
    if isinstance(x, (int,float)) and not pd.isna(x):
        return bool(x)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to summary_w2.csv")
    ap.add_argument("--rc-threshold", type=float, default=2.0, help="two-step 門檻（預設 2.0）")
    ap.add_argument("--fail-on-warn", action="store_true", help="有任何問題直接以非零碼結束")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        print(f"[validate] 缺少欄位: {missing}", file=sys.stderr)
        sys.exit(2)

    problems = []
    for idx, row in df.iterrows():
        y = str(row.get("year","?"))

        fi_min, fi_max = to_f(row["fi_min"]), to_f(row["fi_max"])
        cv_min, cv_max = to_f(row["cv_min"]), to_f(row["cv_max"])

        # --- FI / CV 區間檢查：min ≤ max ---
        if fi_min is not None and fi_max is not None and fi_min > fi_max:
            problems.append(f"Year={y}: [FI] fi_min({fi_min}) > fi_max({fi_max}) → 應為 min ≤ max")
        if cv_min is not None and cv_max is not None and cv_min > cv_max:
            problems.append(f"Year={y}: [CV] cv_min({cv_min}) > cv_max({cv_max}) → 應為 min ≤ max")

        # 建議合理範圍（非硬性）：FI∈[0,1]；CV∈[0,2]（可調）
        for name, val, lo, hi in [
            ("FI.min", fi_min, 0.0, 1.0),
            ("FI.max", fi_max, 0.0, 1.0),
            ("CV.min", cv_min, 0.0, 2.0),
            ("CV.max", cv_max, 0.0, 2.0),
        ]:
            if val is not None and not (lo <= val <= hi):
                problems.append(f"Year={y}: [{name}] {val} 超出建議範圍[{lo},{hi}]（僅警示）")

        eps_star = to_f(row["eps_star"])
        if eps_star is None:
            problems.append(f"Year={y}: [ε*] 缺失（建議填寫由 ε-sweep 得到的臨界阻尼）")

        lam_star      = to_f(row["lam_star"])
        lam_band_low  = to_f(row["lam_star_band_low"])
        lam_band_high = to_f(row["lam_star_band_high"])

        rc_two_step = to_b(row["rc_two_step"])

        # two-step 規則：未通過則不應有 lam_star 或 band
        if rc_two_step is True:
            # 若有 band，檢查 lam_star 是否落在 band 內（若三者皆存在）
            if (lam_star is not None) and (lam_band_low is not None) and (lam_band_high is not None):
                if not (min(lam_band_low, lam_band_high) <= lam_star <= max(lam_band_low, lam_band_high)):
                    problems.append(f"Year={y}: [λ*] lam_star({lam_star}) 不在 band [{lam_band_low},{lam_band_high}] 內")
        else:
            # 未通過 two-step：若仍填了 lam_*，給出明確警告
            if lam_star is not None or (lam_band_low is not None) or (lam_band_high is not None):
                problems.append(f"Year={y}: [λ*] two-step 未通過，但存在 lam_star/band → 應清空")

        # RC 資訊
        rc_max = to_f(row["rc_max"])
        rc_at  = to_f(row["rc_at"])
        if (rc_max is None) ^ (rc_at is None):
            problems.append(f"Year={y}: [RC] rc_max/rc_at 應成對出現或成對缺失")

        # proxy/measured 標記
        pm = str(row["proxy_measured"]).strip().lower() if not pd.isna(row["proxy_measured"]) else ""
        if pm and pm not in ("proxy","measured"):
            problems.append(f"Year={y}: [mode] proxy_measured＝{pm} 不在 {{proxy, measured}}")

    if problems:
        print("=== PROBLEMS ===")
        for p in problems:
            print("- " + p)
        if args.fail_on_warn:
            sys.exit(1)
    else:
        print("[validate] OK：未發現結構性問題。")

if __name__ == "__main__":
    main()