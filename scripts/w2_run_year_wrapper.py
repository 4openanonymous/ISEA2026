#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, shlex
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--eps-grid", default="0.05,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--lam-grid", default="0.40,0.50,0.60,0.65,0.66,0.67,0.68,0.69,0.70")
    ap.add_argument("--rho-qboundary", type=float, default=0.80)
    ap.add_argument("--rho-qinterior", type=float, default=0.50)
    ap.add_argument("--rc-threshold", type=float, default=2.0)
    ap.add_argument("--runner", default="scripts/w2_run_year.py",
                    help="指向舊版單年執行腳本的位置")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    cmd = (
        f'python {shlex.quote(args.runner)} '
        f'--baseline {shlex.quote(args.baseline)} '
        f'--events {shlex.quote(args.events)} '
        f'--outdir {shlex.quote(args.outdir)} '
        f'--eps-grid "{args.eps_grid}" '
        f'--lam-grid "{args.lam_grid}" '
        f'--rho-qboundary {args.rho_qboundary} '
        f'--rho-qinterior {args.rho_qinterior} '
        f'--rc-threshold {args.rc_threshold}'
    )
    print("[w2_run_year] exec:", cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise SystemExit(ret)

if __name__ == "__main__":
    main()