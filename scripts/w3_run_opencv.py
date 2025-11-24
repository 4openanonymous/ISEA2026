#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W3 OpenCV runner (per-year)
- Input:  reports/W2_runs/<YEAR>/events.csv  (or first *.csv found)
- Output: reports/W3_logs/<YEAR>/events_log_<YEAR>[__NOTE][__YYYYMMDD_HHMMSS].csv
Run:
  python -m scripts.w3_run_opencv --year 2021
  python -m scripts.w3_run_opencv --years 2021,2022 --notes "baseline"
  python -m scripts.w3_run_opencv --year 2021 --notes "batch run" --versioning time
"""
from pathlib import Path
import time, argparse, sys, re, datetime
import pandas as pd
from scripts.w3_logger import DevTimer

REPO  = Path(__file__).resolve().parents[1]
W2_DIR = REPO / "reports" / "W2_runs"
W3_DIR = REPO / "reports" / "W3_logs"

ALIAS = {
    "run_id": ["run_id","runID","run"],
    "id": ["id","event_id"],
    "dt_target": ["Δt_target","dt_target","t_target"]
}

def pick(df: pd.DataFrame, key: str) -> str:
    for name in ALIAS[key]:
        if name in df.columns:
            return name
    raise KeyError(f"Missing required column for {key}. Tried: {ALIAS[key]}")

def find_events_csv(year: str) -> Path:
    ydir = W2_DIR / str(year)
    if not ydir.is_dir():
        raise FileNotFoundError(f"W2 year folder not found: {ydir}")
    cand = ydir / "events.csv"
    if cand.exists():
        return cand
    csvs = sorted(ydir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {ydir}")
    return csvs[0]

def sanitize_note(note: str) -> str:
    if not note:
        return "run"
    s = note.strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "run"

def make_log_path(year: str, notes: str, versioning: str) -> Path:
    out_dir = W3_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    note_slug = sanitize_note(notes)
    if versioning == "plain":
        fname = f"events_log_{year}.csv"
    elif versioning == "note":
        fname = f"events_log_{year}__{note_slug}.csv"
    elif versioning == "time":
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"events_log_{year}__{note_slug}__{ts}.csv"
    else:
        raise ValueError("versioning must be one of: plain, note, time")
    return out_dir / fname

def render_one(delay_sec: float) -> None:
    time.sleep(max(0.0, float(delay_sec)))  # 模擬顯影；之後換實渲染

def run_one_year(year: str, renderer: str, notes: str, versioning: str, overwrite: bool):
    events_path = find_events_csv(year)
    df = pd.read_csv(events_path)

    run_col = pick(df, "run_id")
    id_col  = pick(df, "id")
    tgt_col = pick(df, "dt_target")

    log_path = make_log_path(year, notes, versioning)
    if overwrite and log_path.exists():
        log_path.unlink()

    for _, row in df.iterrows():
        run_id = str(row[run_col])
        ev_id  = str(row[id_col])
        delay  = float(row[tgt_col])
        with DevTimer(run_id=run_id, id_=ev_id, renderer=renderer,
                      log_path=str(log_path), notes=notes):
            render_one(delay)

    print(f"[{year}] wrote {log_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=str, help="Single year, e.g., 2021")
    ap.add_argument("--years", type=str, help="Comma-separated years, e.g., 2021,2022")
    ap.add_argument("--renderer", default="opencv")
    ap.add_argument("--notes", default="")
    ap.add_argument("--versioning", choices=["plain","note","time"], default="note",
                    help="Filename strategy: plain=single file; note=per note; time=per note+timestamp")
    ap.add_argument("--overwrite", action="store_true", help="Remove existing target log before writing")
    args = ap.parse_args()

    years = []
    if args.year:
        years = [args.year]
    if args.years:
        years += [y.strip() for y in args.years.split(",") if y.strip()]
    if not years:
        print("Specify --year 2021 or --years 2021,2022", file=sys.stderr)
        sys.exit(2)

    for y in years:
        run_one_year(y, renderer=args.renderer, notes=args.notes,
                     versioning=args.versioning, overwrite=args.overwrite)

if __name__ == "__main__":
    main()