#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, subprocess, shlex
from pathlib import Path

def run(cmd):
    print("[orchestrate] exec:", cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise SystemExit(ret)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobset", default="configs/w2_jobset.json")
    args = ap.parse_args()

    cfg = json.loads(Path(args.jobset).read_text(encoding="utf-8"))

    baseline   = cfg["baseline"]
    years      = cfg["years"]
    events_glob= cfg["events_glob"]         # e.g. "reports/W2_runs/{year}/events.csv"
    out_scan   = cfg["outdir_scans"]        # e.g. "reports/W2_scans/{year}"
    out_merge  = cfg["outdir_merged"]       # e.g. "reports/W2_combined"

    eps_grid   = ",".join(cfg.get("eps_grid", []))
    lam_grid   = ",".join(cfg.get("lam_grid", []))
    qb         = cfg.get("rho_qboundary", 0.80)
    qi         = cfg.get("rho_qinterior", 0.50)
    thr        = cfg.get("rc_threshold", 2.0)

    # 逐年跑
    for y in years:
        events = events_glob.format(year=y)
        outdir = out_scan.format(year=y)
        Path(outdir).mkdir(parents=True, exist_ok=True)

        cmd = (
            f'python scripts/w2_run_year.py '
            f'--baseline {shlex.quote(baseline)} '
            f'--events {shlex.quote(events)} '
            f'--outdir {shlex.quote(outdir)} '
            f'--eps-grid "{eps_grid}" '
            f'--lam-grid "{lam_grid}" '
            f'--rho-qboundary {qb} '
            f'--rho-qinterior {qi} '
            f'--rc-threshold {thr}'
        )
        run(cmd)

    # 合併
    Path(out_merge).mkdir(parents=True, exist_ok=True)
    m_cmd = (
        f'python scripts/w2_merge_reports.py '
        f'--root {shlex.quote(Path(out_scan).parent.as_posix()) if "{year}" in out_scan else shlex.quote(Path(out_scan).parent.as_posix())} '
        f'--outdir {shlex.quote(out_merge)} '
        f'--sanitize --validate'
    )
    # 如果 out_scan 用了 {year}，其 parent 即 W2_scans/
    run(m_cmd)

if __name__ == "__main__":
    main()