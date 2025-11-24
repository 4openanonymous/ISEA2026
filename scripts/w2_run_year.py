#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def ema_by_order(x, order_idx, epsilon):
    x = np.asarray(x, dtype=float)
    ord_idx = np.argsort(order_idx)
    inv_ord = np.empty_like(ord_idx)
    inv_ord[ord_idx] = np.arange(len(order_idx))
    xs = x[ord_idx]
    out = []
    prev = None
    for v in xs:
        prev = v if prev is None else (1.0 - epsilon)*v + epsilon*prev
        out.append(prev)
    out = np.asarray(out)[inv_ord]
    return out

def dt_prime(d_hat, rho, t_min, t_max, gamma, epsilon, lam, order_idx):
    dt = t_min + (t_max - t_min) * np.power(d_hat, gamma)
    dt_s = ema_by_order(dt, order_idx, epsilon)
    if rho is None:
        rho = np.zeros_like(d_hat, dtype=float)
    return dt_s * (1.0 + lam * rho)

def flicker_index(series):
    x = np.asarray(series, dtype=float)
    if len(x) < 2: return np.nan
    return float(np.sum(np.abs(np.diff(x))) / (np.sum(np.abs(x)) + 1e-12))

def cv_first_diff(series):
    x = np.asarray(series, dtype=float)
    if len(x) < 2: return np.nan
    dv = np.diff(x)
    mu = float(np.mean(x))
    return float(np.std(dv, ddof=1) / (abs(mu) + 1e-12))

def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2: return np.nan
    xm, ym = x - x.mean(), y - y.mean()
    den = (np.sqrt((xm*xm).sum()) * np.sqrt((ym*ym).sum()) + 1e-12)
    return float((xm*ym).sum() / den)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100.0)

def cohens_d(x_b, x_i):
    x_b = np.asarray(x_b, dtype=float); x_i = np.asarray(x_i, dtype=float)
    nb, ni = len(x_b), len(x_i)
    if nb < 2 or ni < 2: return np.nan
    sb2 = float(np.var(x_b, ddof=1)); si2 = float(np.var(x_i, ddof=1))
    sp = np.sqrt(((nb-1)*sb2 + (ni-1)*si2) / (nb + ni - 2 + 1e-12))
    return float((np.mean(x_b) - np.mean(x_i)) / (sp + 1e-12))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="path to W2_baseline.json")
    ap.add_argument("--events",   required=True, help="path to events.csv")
    ap.add_argument("--outdir",   default="reports/W2_perceptual", help="output dir")
    # 掃描網格（可依需要調整）
    ap.add_argument("--eps-grid", default="0.0,0.1,0.2,0.3,0.4,0.6", help="epsilon grid (comma separated)")
    ap.add_argument("--lam-grid", default="0.1,0.3,0.5,0.8,1.0,1.5,2.0", help="lambda grid (comma separated)")
    ap.add_argument("--rho-qboundary", type=float, default=0.80, help="邊界分位數（例如 0.80）")
    ap.add_argument("--rho-qinterior", type=float, default=0.50, help="內部分位數（例如 0.50）")
    ap.add_argument("--rc-threshold", type=float, default=2.0, help="RC 門檻（例如 2.0）")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    # 固定 γ；ε/λ 由掃描決定
    gamma = float(cfg["gamma"])
    t_min = float(cfg["t_min"]); t_max = float(cfg["t_max"])

    df = pd.read_csv(args.events)
    # 基本欄位
    needed = ["d_hat","order_idx"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"events.csv 缺少欄位：{c}")
    d_hat = df["d_hat"].astype(float).to_numpy()
    order_idx = df["order_idx"].astype(int).to_numpy()

    rho = df["rho"].astype(float).to_numpy() if "rho" in df.columns else None
    # 優先使用 dt_applied；若缺/全空，改用 dt_target（標示為 proxy）
    use_proxy = True
    if "dt_applied" in df.columns and df["dt_applied"].notna().any():
        y_obs = df["dt_applied"].astype(float).to_numpy()
        use_proxy = False
    elif "dt_target" in df.columns:
        y_obs = df["dt_target"].astype(float).to_numpy()
        use_proxy = True
    else:
        raise ValueError("events.csv 需包含 dt_applied 或 dt_target")

    # 1) ε 掃描（固定 λ = baseline）
    lam0 = float(cfg["lambda"])
    eps_list = [float(x) for x in args.eps_grid.split(",")]
    rows_eps = []
    for e in eps_list:
        y_pred = dt_prime(d_hat, rho, t_min, t_max, gamma, e, lam0, order_idx)
        rows_eps.append(dict(
            epsilon=e,
            FI=flicker_index(y_pred),
            CV_v=cv_first_diff(y_pred),
            r_dhat=pearson(d_hat, y_pred),
            r_obs=pearson(y_obs, y_pred),
            MAPE=mape(y_obs, y_pred),
        ))
    df_eps = pd.DataFrame(rows_eps).sort_values("epsilon")
    # 圖：FI vs ε
    plt.figure(figsize=(5.2,4.2))
    plt.plot(df_eps["epsilon"], df_eps["FI"], marker="o")
    plt.xlabel("epsilon"); plt.ylabel("Flicker Index"); plt.title("FI vs epsilon")
    plt.tight_layout(); plt.savefig(outdir/"fi_vs_epsilon.png", dpi=300); plt.close()
    # 圖：CV_v vs ε
    plt.figure(figsize=(5.2,4.2))
    plt.plot(df_eps["epsilon"], df_eps["CV_v"], marker="o")
    plt.xlabel("epsilon"); plt.ylabel("CV(Δv)"); plt.title("CV(Δv) vs epsilon")
    plt.tight_layout(); plt.savefig(outdir/"cv_v_vs_epsilon.png", dpi=300); plt.close()

    # 2) λ 掃描（固定 ε = baseline），若無 rho 則略過
    lam_list = [float(x) for x in args.lam_grid.split(",")]
    df_lam = None
    if rho is not None:
        e0 = float(cfg["epsilon"])
        # 以 rho 分位劃分邊界/內部
        qB = np.quantile(rho, args.rho_qboundary)
        qI = np.quantile(rho, args.rho_qinterior)
        B_mask = rho >= qB
        I_mask = rho <= qI
        rows_lam = []
        for L in lam_list:
            yL = dt_prime(d_hat, rho, t_min, t_max, gamma, e0, L, order_idx)
            if B_mask.sum() >= 5 and I_mask.sum() >= 5:
                RC = float(np.mean(yL[B_mask]) / (np.mean(yL[I_mask]) + 1e-12))
                d_eff = cohens_d(yL[B_mask], yL[I_mask])
            else:
                RC, d_eff = np.nan, np.nan
            rows_lam.append(dict(lambda_=L, RC=RC, cohens_d=d_eff))
        df_lam = pd.DataFrame(rows_lam).sort_values("lambda_")
        # 圖：RC vs λ
        plt.figure(figsize=(5.2,4.2))
        plt.plot(df_lam["lambda_"], df_lam["RC"], marker="o")
        plt.axhline(args.rc_threshold, ls="--")
        plt.xlabel("lambda"); plt.ylabel("RC = mean(dt’|B) / mean(dt’|I)")
        plt.title("Resistance Contrast vs lambda")
        plt.tight_layout(); plt.savefig(outdir/"rc_vs_lambda.png", dpi=300); plt.close()

    # 3) 報告
    md = outdir / "W2_perceptual_report.md"
    lines = []
    lines.append("# W2 Perceptual Verification\n")
    lines.append(f"- Source: `{args.events}`\n- Baseline: `{args.baseline}`\n")
    lines.append("## Respiratory Smoothness (ε sweep, λ fixed)\n")
    lines.append(df_eps.to_markdown(index=False))
    lines.append("\n**Figures:** `fi_vs_epsilon.png`, `cv_v_vs_epsilon.png`\n")
    if df_lam is None:
        lines.append("## Perceptual Threshold (λ sweep)\n")
        lines.append("_Skipped_ (ρ not available). Provide `rho` to enable RC(λ) analysis.\n")
    else:
        lines.append("## Perceptual Threshold (λ sweep, ε fixed)\n")
        lines.append(df_lam.to_markdown(index=False))
        # 找臨界 λ*
        thr = args.rc_threshold
        lam_star = None
        vals = df_lam["RC"].to_numpy()
        Ls   = df_lam["lambda_"].to_numpy()
        for i in range(len(vals)-1):
            if vals[i] >= thr and vals[i+1] >= thr:
                lam_star = Ls[i]
                break
        lines.append(f"\n- Threshold τ = {thr:.2f}; λ* = {('N/A' if lam_star is None else f'{lam_star:.3g}')} (two-step rule)\n")
        lines.append("**Figure:** `rc_vs_lambda.png`\n")
    lines.append("\n---\n")
    lines.append(f"_Observed series used: {'dt_target (proxy)' if use_proxy else 'dt_applied (measured)'}_")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {md}")
    print(f"[save] {outdir/'fi_vs_epsilon.png'}")
    print(f"[save] {outdir/'cv_v_vs_epsilon.png'}")
    if df_lam is not None:
        print(f"[save] {outdir/'rc_vs_lambda.png'}")

if __name__ == "__main__":
    main()