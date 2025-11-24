#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W4-1 Perceptual Rhythm Renderer (OpenCV)
- Reads merged CSV (e.g., reports/W3_logs/2021/events_merged_2021__baseline-run.csv)
- Maps Δt (dt_applied or dt_model) -> temporal rhythm (frame repetition)
- Maps ρ (rho) -> brightness (or flicker on:off ratio)

Usage examples:

  # 基準：以 dt_applied，規一到 60 秒，亮度模式，顯示文字
  python -m scripts.w4_render_driver \
    --csv reports/W3_logs/2021/events_merged_2021__baseline-run.csv \
    --dt-col dt_applied --auto-seconds 60 \
    --out out/W4_REF_2021.mp4 --fps 30 --mode brightness --text

  # 以 dt_model 驅動，ρ 做 5-95 分位裁剪並套 1.2 gamma
  python -m scripts.w4_render_driver \
    --csv reports/W4_variants/2021/g1.2_e0.3_l0.7/events_dt_model.csv \
    --dt-col dt_model --auto-seconds 60 \
    --rho-clip 5,95 --rho-gamma 1.2 \
    --out out/W4_variant.mp4 --mode brightness --text

  # flicker 模式 + 邊界 LED + 微噪聲
  python -m scripts.w4_render_driver \
    --csv reports/W3_logs/2021/events_merged_2021__baseline-run.csv \
    --dt-col dt_applied --auto-seconds 60 \
    --mode flicker --led --led-quantile 90 --noise 3 \
    --out out/W4_REF_2021_flicker.mp4
"""
from pathlib import Path
import argparse, re
import numpy as np
import pandas as pd
import cv2

ALIAS = {
    "run_id":     ["run_id","runid","run"],
    "id":         ["id","event_id"],
    "d_hat":      ["d_hat","dhat","d̂","d-hat"],
    "rho":        ["rho","ρ","rho_i"],
    "dt_applied": ["Δt_applied","dt_applied","dt","dt_applied(s)","delta_t_applied","dt_applied_y"],
}

def _pick(df, key):
    cols = {c.strip().lower(): c for c in df.columns}
    for k in ALIAS[key]:
        k2 = k.lower()
        if k2 in cols:
            return cols[k2]
    raise KeyError(f"Missing column for {key}. Have={list(df.columns)} Accept={ALIAS[key]}")

def _normalize_series(x, pclip=(1,99), gamma=1.0):
    x = pd.to_numeric(x, errors="coerce")
    lo = np.nanpercentile(x, pclip[0])
    hi = np.nanpercentile(x, pclip[1])
    y = (x - lo) / (hi - lo + 1e-12)
    y = y.clip(0, 1)
    if gamma != 1.0:
        y = np.power(y, float(gamma))
    return y, (float(lo), float(hi))

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _parse_clip(s: str):
    try:
        a, b = s.split(",")
        return (float(a), float(b))
    except Exception:
        raise ValueError("--rho-clip must be 'lo,hi', e.g., '1,99' or '5,95'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV path (merged or dt_model variant)")
    ap.add_argument("--out", required=True, help="output mp4 or a directory for frames")
    ap.add_argument("--fps", type=int, default=30, help="base fps (frame repetition approximates Δt)")
    ap.add_argument("--speed", type=float, default=1.0, help="global time scale (Δt / speed)")
    ap.add_argument("--auto-seconds", type=float, default=0.0,
                    help="automatically rescale time so full clip ≈ given seconds (0=off)")
    ap.add_argument("--mode", choices=["brightness","flicker"], default="brightness",
                    help="brightness: rho->gray; flicker: rho drives on/off ratio")
    ap.add_argument("--size", default="960x540", help="WxH, e.g., 1280x720")
    ap.add_argument("--max-frames-per-event", type=int, default=120,
                    help="cap to avoid very long events (default ~4s @30fps)")
    ap.add_argument("--text", action="store_true", help="overlay id & Δt")
    ap.add_argument("--font-scale", type=float, default=0.6)
    ap.add_argument("--thickness", type=int, default=1)
    # 可選時間欄位與 rho 映射參數
    ap.add_argument("--dt-col", default="dt_applied",
                    help="time column to drive rhythm, e.g., dt_applied or dt_model")
    ap.add_argument("--rho-clip", default="1,99",
                    help="percentile clipping for rho, e.g., '1,99' or '5,95'")
    ap.add_argument("--rho-gamma", type=float, default=1.0,
                    help="gamma curve for rho mapping (>=0). 1.0 is linear")
    # 額外視覺提示（可選）
    ap.add_argument("--led", action="store_true", help="corner LED when rho above quantile")
    ap.add_argument("--led-quantile", type=float, default=90.0, help="LED threshold quantile of rho_norm (0–100)")
    ap.add_argument("--noise", type=float, default=0.0, help="std of gaussian noise added to brightness (0=off)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # 畫面大小
    m = re.match(r"(\d+)[xX](\d+)", args.size)
    if not m:
        raise ValueError("--size must be <W>x<H>")
    W, H = int(m.group(1)), int(m.group(2))

    # 欄位
    col_id  = _pick(df, "id")
    col_rho = _pick(df, "rho")

    # 時間欄位：優先用 --dt-col，若不存在則退回別名
    dt_col = args.dt_col if args.dt_col in df.columns else _pick(df, "dt_applied")

    # ρ 正規化（百分位裁剪 + gamma）
    clip_lo, clip_hi = _parse_clip(args.rho_clip)
    rho_norm, (rho_lo, rho_hi) = _normalize_series(df[col_rho], pclip=(clip_lo, clip_hi), gamma=args.rho_gamma)

    # 取 Δt
    dt_raw = pd.to_numeric(df[dt_col], errors="coerce").fillna(0).clip(lower=0)
    if (dt_raw <= 0).all():
        raise ValueError(f"All values in '{dt_col}' <= 0 or NaN. Check your CSV.")

    # 自動規一總時長：以 sum(dt_raw)/auto_seconds 當 speed
    speed = float(args.speed)
    if getattr(args, "auto_seconds", 0) > 0:
        total = float(dt_raw.sum())
        # 防止除零
        target = max(1e-6, float(args.auto_seconds))
        speed = max(1e-6, total / target)

    # 輸出器
    outp = Path(args.out)
    if outp.suffix.lower() in [".mp4", ".mov", ".m4v", ".avi"]:
        _ensure_dir(outp)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(outp), fourcc, args.fps, (W, H))
        write_frame = lambda img: writer.write(img)
        finalize = lambda: writer.release()
    else:
        _ensure_dir(outp / "frame_000001.png")
        counter = {"i": 0}
        def write_frame(img):
            counter["i"] += 1
            cv2.imwrite(str(outp / f"frame_{counter['i']:06d}.png"), img)
        finalize = lambda: None

    # 取 LED 門檻
    led_thr = None
    if args.led:
        q = float(np.clip(args.led_quantile, 0, 100))
        led_thr = float(np.nanpercentile(rho_norm.values, q))

    base_bg = np.zeros((H, W, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)  # 固定種子，審計友好
    total_frames, used_seconds = 0, 0.0

    for i, row in df.iterrows():
        rid = str(row[col_id])
        r   = float(rho_norm.iloc[i]) if not pd.isna(rho_norm.iloc[i]) else 0.0
        dts = float(dt_raw.iloc[i]) / max(speed, 1e-6)
        n_frames = max(1, min(args.max_frames_per_event, int(round(dts * args.fps))))

        # brightness 或 flicker 的底值
        if args.mode == "brightness":
            val = int(round(20 + 220 * r))
            frame = base_bg.copy()
            frame[:, :] = (val, val, val)
        else:
            on_val  = int(round(40 + 215 * r))
            off_val = 10
            on_frames  = max(1, int(round(n_frames * (0.3 + 0.7 * r))))  # 30%~100% on
            off_frames = max(0, n_frames - on_frames)

        # 噪聲（微材質）
        def add_noise(img):
            if args.noise <= 0: return img
            noise = rng.normal(0, args.noise, size=img.shape).astype(np.float32)
            out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return out

        # 角落 LED（ρ 高於分位）
        def draw_led(img):
            if not args.led: return img
            thr = led_thr if led_thr is not None else 1.0  # 極端情況
            if r >= thr:
                # 右上角 12x12 小方塊
                img[8:20, W-20:W-8] = (255, 255, 255)
            return img

        # 疊字
        def draw_text(img):
            # 僅在啟用時顯示
            return img if not args.text else cv2.putText(
                img, f"id={rid}  Δt={dts:.3f}s  ρ~{r:.2f}  [{dt_col}]",
                (16, H - 24), cv2.FONT_HERSHEY_SIMPLEX,
                args.font_scale, (255, 255, 255), args.thickness, cv2.LINE_AA
            )

        if args.mode == "brightness":
            for _ in range(n_frames):
                img = draw_text(draw_led(add_noise(frame.copy())))
                write_frame(img)
        else:
            frame_on  = np.full((H, W, 3), on_val,  np.uint8)
            frame_off = np.full((H, W, 3), off_val, np.uint8)
            for _ in range(on_frames):
                img = draw_text(draw_led(add_noise(frame_on.copy())))
                write_frame(img)
            for _ in range(off_frames):
                img = draw_text(draw_led(add_noise(frame_off.copy())))
                write_frame(img)

        total_frames += n_frames
        used_seconds += dts

    finalize()
    seconds = total_frames / float(args.fps)
    print(f"[done] wrote ~{total_frames} frames (~{seconds:.1f}s) to {outp}")
    print(f"[info] dt_total_raw={dt_raw.sum():.3f}s  speed={speed:.6f}  used_seconds={used_seconds:.3f}s  rho_clip=({clip_lo},{clip_hi}) rho_gamma={args.rho_gamma}")
    if args.led:
        print(f"[info] LED quantile={args.led_quantile:.1f} -> thr≈{led_thr:.3f}")

if __name__ == "__main__":
    main()