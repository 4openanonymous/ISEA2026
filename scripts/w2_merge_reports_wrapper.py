#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, shlex, sys, re, shutil, subprocess, json
from pathlib import Path
import pandas as pd

REPORT_BASENAME = "W2_perceptual_report_{year}.md"

COLUMNS = [
    "year",
    "fi_min","fi_max",
    "cv_min","cv_max",
    "eps_star",
    "lam_star_band_low","lam_star_band_high","lam_star",
    "rc_max","rc_at","rc_two_step",
    "proxy_measured",
    "rc_series",
    "path"
]

# ---------- 解析器（從各年 .md 抽欄位） ----------
RE_EPS_STAR         = re.compile(r"(?:ε\*|epsilon\*)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
RE_LAM_STAR_SINGLE  = re.compile(r"(?:λ\*|lambda\*)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
RE_LAM_BAND         = re.compile(r"(?:λ\*|lambda\*)[^0-9\[]*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]", re.I)
RE_FI_RANGE         = re.compile(r"FI[^0-9\[]*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]", re.I)
RE_CV_RANGE         = re.compile(r"CV\s*\(\s*Δv\s*\)[^0-9\[]*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]", re.I)
RE_RC_MAX_AT        = re.compile(r"(?:RC[_ ]?max|max\s*RC)\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*@\s*([0-9]*\.?[0-9]+)", re.I)
RE_TWO_STEP         = re.compile(r"two[- ]?step\s*[:=]\s*(pass|fail)", re.I)
RE_PROXY_MEASURED   = re.compile(r"(proxy|measured)", re.I)
RE_RC_SERIES        = re.compile(r"RC\(\s*λ\s*\)\s*series\s*[:：]\s*([0-9\.\:\,\s]+)", re.I)

def safe_float(x):
    try: return float(x)
    except: return None

def parse_years(s: str|None):
    if not s: return None
    yrs = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        if not re.fullmatch(r"\d{4}", tok):
            raise ValueError(f"--years 內容不合法: {tok}")
        yrs.append(tok)
    return yrs

def collect_reports(root: Path, years: list[str]|None, strict: bool) -> dict:
    picked = {}
    if years:
        for y in years:
            p = root / y / REPORT_BASENAME.format(year=y)
            if not p.exists():
                msg = f"[w2_merge_reports] MISSING for {y}: {p}"
                if strict: raise FileNotFoundError(msg)
                print("WARN:", msg, file=sys.stderr); continue
            picked[y] = p
    else:
        for d in sorted(root.glob("*")):
            if not d.is_dir(): continue
            y = d.name
            if not re.fullmatch(r"\d{4}", y): continue
            p = d / REPORT_BASENAME.format(year=y)
            if p.exists(): picked[y] = p
    return picked

def stage_reports(picked: dict[str, Path], stage_root: Path):
    if stage_root.exists(): shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    for y, src in picked.items():
        dst_dir = stage_root / y
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        for img in src.parent.glob("*.png"):
            shutil.copy2(img, dst_dir / img.name)
    return stage_root

def parse_report(md_path: Path) -> dict:
    y = md_path.stem.split("_")[-1]
    txt = md_path.read_text(encoding="utf-8", errors="ignore")
    out = {c: None for c in COLUMNS}
    out.update({"year": y, "path": str(md_path)})

    m = RE_EPS_STAR.search(txt)
    if m: out["eps_star"] = safe_float(m.group(1))

    m = RE_LAM_BAND.search(txt)
    if m:
        out["lam_star_band_low"]  = safe_float(m.group(1))
        out["lam_star_band_high"] = safe_float(m.group(2))
    m = RE_LAM_STAR_SINGLE.search(txt)
    if m: out["lam_star"] = safe_float(m.group(1))

    m = RE_FI_RANGE.search(txt)
    if m:
        lo, hi = safe_float(m.group(1)), safe_float(m.group(2))
        if lo is not None and hi is not None:
            out["fi_min"], out["fi_max"] = min(lo,hi), max(lo,hi)

    m = RE_CV_RANGE.search(txt)
    if m:
        lo, hi = safe_float(m.group(1)), safe_float(m.group(2))
        if lo is not None and hi is not None:
            out["cv_min"], out["cv_max"] = min(lo,hi), max(lo,hi)

    m = RE_RC_MAX_AT.search(txt)
    if m:
        out["rc_max"] = safe_float(m.group(1))
        out["rc_at"]  = safe_float(m.group(2))

    m = RE_TWO_STEP.search(txt)
    if m: out["rc_two_step"] = True if m.group(1).lower()=="pass" else False

    m = RE_PROXY_MEASURED.search(txt)
    if m: out["proxy_measured"] = m.group(1).lower()

    m = RE_RC_SERIES.search(txt)
    if m: out["rc_series"] = re.sub(r"\s+", "", m.group(1))
    return out

def write_summary(outdir: Path, rows: list[dict]):
    df = pd.DataFrame(rows, columns=COLUMNS)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "summary_w2.csv"
    md_path  = outdir / "summary_w2.md"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    def fmt(x): 
        return "NA" if x is None or (isinstance(x,float) and pd.isna(x)) else x
    headers = "| " + " | ".join(COLUMNS) + " |"
    sep     = "| " + " | ".join(["---"]*len(COLUMNS)) + " |"
    lines = ["# W2 Summary (auto-collected)", "", headers, sep]
    for _, r in df.iterrows():
        vals = [str(fmt(r[c])) for c in COLUMNS]
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path

def write_results_section(outdir: Path, rows: list[dict]):
    def fmt(x):
        return "NA" if x is None or (isinstance(x,float) and pd.isna(x)) else x
    lines = ["# Results (W2 Cross-Year Summary)", ""]
    for r in sorted(rows, key=lambda z: z["year"]):
        y = r["year"]
        fi_rng = f"[{fmt(r['fi_min'])}, {fmt(r['fi_max'])}]"
        cv_rng = f"[{fmt(r['cv_min'])}, {fmt(r['cv_max'])}]"
        lam_band = f"[{fmt(r['lam_star_band_low'])}, {fmt(r['lam_star_band_high'])}]"
        lam_star = fmt(r["lam_star"])
        eps_star = fmt(r["eps_star"])
        two_step = fmt(r["rc_two_step"])
        rcmax, rcat = fmt(r["rc_max"]), fmt(r["rc_at"])
        prox = fmt(r["proxy_measured"])
        lines.append(
            f"**{y}.** FI={fi_rng}; CV(Δv)={cv_rng}; ε*={eps_star}; λ*={lam_star}（band={lam_band}, two-step={two_step}）; "
            f"RC_max={rcmax}@λ={rcat}; mode={prox}."
        )
    (outdir / "results_section.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    required=True, help="掃描各年報告的根目錄，如 reports/W2_scans")
    ap.add_argument("--outdir",  required=True, help="合併輸出的資料夾，如 reports/W2_combined")
    ap.add_argument("--years", type=str, default=None, help='年份白名單，例如 "2020,2021,2022,2023,2024"')
    ap.add_argument("--strict", action="store_true", help="缺年份或缺檔即報錯")
    ap.add_argument("--dry-run", action="store_true", help="只列清單，不執行合併")
    ap.add_argument("--sanitize", action="store_true", help="two-step 未過→清空 λ* 與 band（後處理）")
    ap.add_argument("--validate", action="store_true", help="合併後跑 validator（後處理）")
    args = ap.parse_args()

    root   = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    years = parse_years(args.years)
    picked = collect_reports(root, years, args.strict)
    if not picked:
        raise SystemExit("[w2_merge_reports] 找不到任何可合併的年報告，請確認目錄或檔名。")

    print("[w2_merge_reports] 將合併以下檔案：")
    for y in sorted(picked):
        print(f"  - {y}: {picked[y]}")

    if args.dry_run:
        print("[w2_merge_reports] dry-run 結束（未執行合併）。")
        return

    # 1) 建立乾淨 staging（隔離，避免吃到 refine/舊檔）
    stage_root = outdir / "_merge_stage"
    if stage_root.exists(): shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    for y, src in picked.items():
        dst_dir = stage_root / y
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        for img in src.parent.glob("*.png"):
            shutil.copy2(img, dst_dir / img.name)

    # 2) 解析 staging 內各年 .md → rows
    rows = []
    for year_dir in sorted(stage_root.glob("*")):
        if not year_dir.is_dir(): continue
        y = year_dir.name
        md = year_dir / f"W2_perceptual_report_{y}.md"
        if not md.exists():
            print(f"[collect] skip {y}: missing {md}", file=sys.stderr)
            continue
        rows.append(parse_report(md))

    if not rows:
        print("[collect] no reports found in stage root.", file=sys.stderr)
        sys.exit(1)

    # 3) 寫 summary 與 results 段落
    csv_path, md_path = write_summary(outdir, rows)
    write_results_section(outdir, rows)
    print(f"[collect] wrote: {csv_path}")
    print(f"[collect] wrote: {md_path}")
    print(f"[collect] wrote: {outdir/'results_section.md'}")

    # 4) 可選後處理（沿用你的既有 sanitize/validate 腳本）
    if args.sanitize:
        s_cmd = f'python scripts/w2_sanitize_summary.py --csv {shlex.quote(str(csv_path))}'
        print("[w2_merge_reports] sanitize:", s_cmd)
        subprocess.call(s_cmd, shell=True)

    if args.validate:
        v_cmd = f'python scripts/w2_validate_summary.py --csv {shlex.quote(str(csv_path))}'
        print("[w2_merge_reports] validate:", v_cmd)
        subprocess.call(v_cmd, shell=True)

if __name__ == "__main__":
    main()