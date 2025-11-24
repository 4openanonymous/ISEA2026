#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W3 verifier (contract-first, audit-friendly)

Contract:
- W2 must NOT contain dt_applied; if present, rename to dt_applied_w2 during preflight.
- W3 log provides dt_applied (instrument-measured).

Inputs:
  reports/W2_runs/<YEAR>/(events.csv or first *.csv)
  reports/W3_logs/<YEAR>/events_log_<YEAR>[__NOTE][__TS].csv  (or --log)
Outputs:
  reports/W3_logs/<YEAR>/events_cleaned_<YEAR>.csv        (W2 after sanitizer)
  reports/W3_logs/<YEAR>/events_merged_<YEAR>[__TAG].csv
  reports/W3_logs/<YEAR>/W3_instrumental_report_<YEAR>[__TAG].md
"""
from pathlib import Path
import argparse, sys, re, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr

REPO  = Path(__file__).resolve().parents[1]
W2_DIR = REPO / "reports" / "W2_runs"
W3_DIR = REPO / "reports" / "W3_logs"

# ----- column aliases -----
ALIAS = {
    "run_id":     ["run_id","runid","run"],
    "id":         ["id","event_id"],
    "d_hat":      ["d_hat","dhat","d̂","d-hat"],
    "rho":        ["rho","ρ","rho_i"],
    "dt_target":  ["Δt_target","dt_target","t_target","delta_t_target"],
    "dt_applied": ["Δt_applied","dt_applied","dt","dt_applied(s)","delta_t_applied"]
}

# ===== helpers =====
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str)
                  .str.replace("\ufeff","", regex=False)
                  .str.strip()
                  .str.lower())
    return df

def pick(df: pd.DataFrame, key: str) -> str:
    cols = set(df.columns)
    for k in ALIAS[key]:
        kk = k.lower()
        if kk in cols: return kk
    raise KeyError(f"Missing required column for {key}. "
                   f"Have: {list(df.columns)} ; Accept: {ALIAS[key]}")

def _find_w2_events(year: str) -> Path:
    ydir = W2_DIR / year
    cand = ydir / "events.csv"
    if cand.exists(): return cand
    csvs = sorted(ydir.glob("*.csv"))
    if not csvs: raise FileNotFoundError(f"No W2 CSV for {year} under {ydir}")
    return csvs[0]

def _sanitize_note(s: str) -> str:
    if not s: return ""
    s = s.strip().lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9_\-]+", "", s)

def _auto_pick_log(year: str, notes: str|None) -> Path:
    ydir = W3_DIR / year
    cands = sorted(ydir.glob(f"events_log_{year}*.csv"))
    if not cands:
        raise FileNotFoundError(f"No W3 log for {year} under {ydir}")
    if notes:
        key = _sanitize_note(notes)
        with_note = [p for p in cands if f"__{key}" in p.stem]
        if with_note: return with_note[-1]
    return cands[-1]

# ===== preflight sanitizer for W2 =====
def sanitize_w2(year: str) -> tuple[pd.DataFrame, Path, dict]:
    """Load W2 events, normalize columns, enforce contract, write cleaned snapshot."""
    raw_path = _find_w2_events(year)
    ev = norm_cols(pd.read_csv(raw_path))
    changes = {"renamed": {}, "dropped": [], "notes": []}

    # required
    run_col = pick(ev, "run_id")
    id_col  = pick(ev, "id")
    dh_col  = pick(ev, "d_hat")
    rh_col  = pick(ev, "rho")
    dt_col  = pick(ev, "dt_target")

    # forbidden-in-W2: dt_applied  -> rename to dt_applied_w2
    if any(c in ev.columns for c in [c.lower() for c in ALIAS["dt_applied"]]):
        # 找到實際命中的那個欄名
        hit = None
        for c in ALIAS["dt_applied"]:
            c = c.lower()
            if c in ev.columns:
                hit = c; break
        ev = ev.rename(columns={hit: "dt_applied_w2"})
        changes["renamed"][hit] = "dt_applied_w2"
        changes["notes"].append("W2 contained forbidden dt_applied; renamed to dt_applied_w2")

    # normalize canonical names
    ev = ev.rename(columns={
        run_col:"run_id", id_col:"id", dh_col:"d_hat", rh_col:"rho", dt_col:"dt_target"
    })
    # enforce numeric types
    for c in ["d_hat","rho","dt_target"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")

    # write cleaned snapshot for audit
    out_dir = W3_DIR / year
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / f"events_cleaned_{year}.csv"
    ev.to_csv(cleaned_path, index=False)
    meta = {"raw": str(raw_path), "cleaned": str(cleaned_path), "changes": changes}
    (out_dir / f"events_cleaned_{year}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ev, cleaned_path, changes

def load_log(log_path: Path, notes_filter: str|None) -> pd.DataFrame:
    lg = norm_cols(pd.read_csv(log_path))
    run_col = pick(lg, "run_id")
    id_col  = pick(lg, "id")
    da_col  = pick(lg, "dt_applied")
    lg = lg.rename(columns={run_col:"run_id", id_col:"id", da_col:"dt_applied"})
    lg["dt_applied"] = pd.to_numeric(lg["dt_applied"], errors="coerce")
    if notes_filter and "notes" in lg.columns:
        key = notes_filter.strip().lower()
        lg = lg[lg["notes"].astype(str).str.lower().str.contains(key, na=False)]
    if lg.empty:
        raise ValueError(f"Filtered log is empty. Check --notes or content: {log_path}")
    return lg

def compute_metrics(df: pd.DataFrame):
    need = ["d_hat","dt_applied"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Merged dataframe missing columns: {miss}. Available: {list(df.columns)}")
    sub = df.dropna(subset=need)
    r_d = np.nan
    if len(sub) > 2:
        r_d, _ = pearsonr(sub["d_hat"], sub["dt_applied"])
    r_t = np.nan
    if "dt_target" in sub and sub["dt_target"].notna().sum() > 2:
        r_t, _ = pearsonr(sub["dt_target"], sub["dt_applied"])
    mape = np.nan
    if "dt_target" in sub:
        nz = sub["dt_target"].replace(0, np.nan)
        mape = (np.abs(sub["dt_applied"] - sub["dt_target"]) / nz).mean() * 100
    return float(r_d), (None if np.isnan(r_t) else float(r_t)), float(mape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True)
    ap.add_argument("--log", help="Explicit path to a W3 events_log CSV")
    ap.add_argument("--notes", help="Fuzzy filter by notes content")
    args = ap.parse_args()

    year = str(args.year)
    log_path = Path(args.log) if args.log else _auto_pick_log(year, args.notes)
    if not log_path.exists():
        raise FileNotFoundError(f"--log not found: {log_path}")

    # 1) preflight sanitize W2
    ev, cleaned_path, changes = sanitize_w2(year)

    # 2) load log
    lg = load_log(log_path, args.notes)

    # 3) merge (no suffix pain because W2 no longer exposes dt_applied)
    merged = pd.merge(ev, lg, on=["run_id","id"], how="inner", validate="one_to_one")

    # 4) outputs
    stem = log_path.stem
    m = re.search(r"events_log_\d{4}__(.+)$", stem)
    tag = f"__{m.group(1)}" if m else f"__{_sanitize_note(args.notes)}" if args.notes else ""

    out_dir = W3_DIR / year
    merged_path = out_dir / f"events_merged_{year}{tag}.csv"
    rpt_path    = out_dir / f"W3_instrumental_report_{year}{tag}.md"
    merged.to_csv(merged_path, index=False)

    # 5) metrics
    n_events  = len(ev)
    n_matched = len(merged)
    comp = 100.0 * n_matched / max(1, n_events)
    r_dhat, r_target, mape = compute_metrics(merged)
    pass_rd   = (not np.isnan(r_dhat)) and (r_dhat >= 0.80)
    pass_mape = (not np.isnan(mape))  and (mape   <= 10.0)
    verdict = "PASS" if (pass_rd and pass_mape) else "FAIL"

    # 6) report with audit notes
    rpt_path.write_text(f"""# W3 Instrumental Verification Report – {year}

**Log file**: `{log_path.name}`
**dt_applied source**: log
**W2 cleaned snapshot**: `{cleaned_path.name}`
**W2 contract fixes**: {json.dumps(changes["changes"] if "changes" in changes else changes, ensure_ascii=False)}

## Matching and Completeness
- Total events: **{n_events}**
- Matched: **{n_matched}**
- Completeness: **{comp:.2f}%**

## Metrics
- r(d̂, Δt_applied) = **{r_dhat:.3f}** {'✅' if pass_rd else '❌'}
- r(Δt_target, Δt_applied) = **{(r_target if r_target is not None else float('nan')):.3f}**
- MAPE(Δt_target, Δt_applied) = **{(mape if not np.isnan(mape) else float('nan')):.2f}%** {'✅' if pass_mape else '❌'}

## Verdict
- Result: **{verdict}**
""", encoding="utf-8")

    print(f"[{year}] cleaned  → {cleaned_path}")
    print(f"[{year}] merged   → {merged_path}")
    print(f"[{year}] report   → {rpt_path}")

if __name__ == "__main__":
    main()