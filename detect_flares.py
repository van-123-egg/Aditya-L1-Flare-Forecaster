"""
Solar Flare Detection Algorithm (Nowcasting) - SoLEXS
=====================================================
Detects and classifies solar flares from SoLEXS lightcurve data.
Processes all available dates and generates a Master Flare Catalog.

This is Milestone 1 + 2 of the hackathon problem statement.
"""
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.ndimage import uniform_filter1d
import os
import glob
import json


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATASET_DIR = "dataset"
SMOOTHING_WINDOW = 30        # seconds - smooth noisy 1s data
BG_WINDOW = 600              # 10-minute rolling median for background
RISE_THRESHOLD = 3.0         # flare start: counts > BG * this factor
MIN_FLARE_DURATION = 30      # minimum flare duration in seconds
MIN_PEAK_COUNTS = 20         # minimum peak counts to be called a flare
MERGE_GAP = 120              # merge detections within 2 minutes


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (handles both flat and nested PRADAN structures)
# ═══════════════════════════════════════════════════════════════
def find_lc_file(folder_path):
    """Find the lightcurve file in a PRADAN download folder."""
    search_roots = [folder_path]
    for item in os.listdir(folder_path):
        nested = os.path.join(folder_path, item)
        if os.path.isdir(nested) and item.startswith("AL1_"):
            search_roots.insert(0, nested)

    for root in search_roots:
        for det in ["SDD2", "SDD1"]:
            sdd_dir = os.path.join(root, det)
            if not os.path.isdir(sdd_dir):
                continue
            lc_files = glob.glob(os.path.join(sdd_dir, "*.lc.gz")) + \
                       glob.glob(os.path.join(sdd_dir, "*.lc"))
            gti_files = glob.glob(os.path.join(sdd_dir, "*.gti.gz")) + \
                        glob.glob(os.path.join(sdd_dir, "*.gti"))
            if lc_files:
                return lc_files[0], gti_files[0] if gti_files else None, det
    return None, None, None


def load_lightcurve(lc_path):
    """Load a SoLEXS lightcurve FITS file into arrays."""
    hdul = fits.open(lc_path)
    data = hdul[1].data
    header = hdul[1].header

    time_raw = data["TIME"].copy()
    counts = data["COUNTS"].copy()

    # Convert to UTC datetime
    mjd_ref = header["MJDREFI"] + header["MJDREFF"]
    tstart = Time(mjd_ref + header["TSTART"] / 86400.0, format="mjd")

    hdul.close()
    return time_raw, counts, tstart, mjd_ref


# ═══════════════════════════════════════════════════════════════
# FLARE DETECTION ALGORITHM
# ═══════════════════════════════════════════════════════════════
def compute_background(counts, window=BG_WINDOW):
    """
    Compute the rolling background using a median filter.
    The background represents the 'quiet Sun' level.
    """
    # Pad edges to avoid boundary effects
    pad = window // 2
    padded = np.pad(counts, pad, mode='edge')

    # Rolling median (robust to flare spikes)
    bg = np.array([
        np.nanmedian(padded[max(0, i - pad):i + pad])
        for i in range(pad, len(padded) - pad)
    ])
    return bg[:len(counts)]


def compute_background_fast(counts, window=BG_WINDOW):
    """
    Fast background estimation using percentile on chunks.
    Much faster than per-element rolling median.
    """
    n = len(counts)
    bg = np.full(n, np.nan)

    # Process in chunks
    chunk_size = window // 2
    for i in range(0, n, chunk_size):
        start = max(0, i - window)
        end = min(n, i + window)
        chunk_median = np.nanpercentile(counts[start:end], 30)  # 30th percentile - below typical flare
        bg[i:min(i + chunk_size, n)] = chunk_median

    # Interpolate any remaining NaNs
    mask = ~np.isnan(bg)
    if mask.sum() > 0:
        bg = np.interp(np.arange(n), np.where(mask)[0], bg[mask])

    return bg


def detect_flares(time_raw, counts, bg, threshold=RISE_THRESHOLD,
                  min_duration=MIN_FLARE_DURATION, min_peak=MIN_PEAK_COUNTS):
    """
    Detect flare events from a lightcurve.

    Algorithm:
    1. Smooth the counts to reduce noise
    2. Find where smoothed counts exceed threshold * background
    3. Group consecutive above-threshold points into events
    4. For each event, find start, peak, and end times
    5. Filter by minimum duration and peak intensity
    """
    # Smooth the counts
    counts_smooth = uniform_filter1d(np.nan_to_num(counts, nan=0), size=SMOOTHING_WINDOW)

    # Threshold: where counts significantly exceed background
    detection_level = bg * threshold
    above = counts_smooth > detection_level

    # Also require absolute minimum
    above = above & (counts_smooth > min_peak / 2)

    # Find contiguous above-threshold regions
    flares = []
    in_flare = False
    start_idx = 0

    for i in range(len(above)):
        if above[i] and not in_flare:
            start_idx = i
            in_flare = True
        elif not above[i] and in_flare:
            # End of a flare candidate
            end_idx = i
            in_flare = False

            # Extend end to where counts return closer to background
            # (the decay tail often dips below threshold briefly)
            while end_idx < len(counts_smooth) - 1 and counts_smooth[end_idx] > bg[end_idx] * 1.5:
                end_idx += 1

            duration = end_idx - start_idx  # in seconds (1s cadence)
            peak_idx = start_idx + np.nanargmax(counts[start_idx:end_idx])
            peak_counts = counts[peak_idx]

            if duration >= min_duration and peak_counts >= min_peak:
                flares.append({
                    "start_idx": start_idx,
                    "peak_idx": peak_idx,
                    "end_idx": end_idx,
                    "start_time": time_raw[start_idx],
                    "peak_time": time_raw[peak_idx],
                    "end_time": time_raw[end_idx - 1],
                    "peak_counts": float(peak_counts),
                    "mean_counts": float(np.nanmean(counts[start_idx:end_idx])),
                    "bg_at_peak": float(bg[peak_idx]),
                    "duration_sec": duration,
                    "rise_time_sec": peak_idx - start_idx,
                    "decay_time_sec": end_idx - 1 - peak_idx,
                    "intensity_ratio": float(peak_counts / max(bg[peak_idx], 1)),
                })

    # If still in a flare at end of data
    if in_flare:
        end_idx = len(counts)
        duration = end_idx - start_idx
        peak_idx = start_idx + np.nanargmax(counts[start_idx:end_idx])
        peak_counts = counts[peak_idx]
        if duration >= min_duration and peak_counts >= min_peak:
            flares.append({
                "start_idx": start_idx,
                "peak_idx": peak_idx,
                "end_idx": end_idx - 1,
                "start_time": time_raw[start_idx],
                "peak_time": time_raw[peak_idx],
                "end_time": time_raw[end_idx - 1],
                "peak_counts": float(peak_counts),
                "mean_counts": float(np.nanmean(counts[start_idx:end_idx])),
                "bg_at_peak": float(bg[peak_idx]),
                "duration_sec": duration,
                "rise_time_sec": peak_idx - start_idx,
                "decay_time_sec": end_idx - 1 - peak_idx,
                "intensity_ratio": float(peak_counts / max(bg[peak_idx], 1)),
            })

    # Merge flares that are close together (within MERGE_GAP seconds)
    merged = []
    for flare in flares:
        if merged and (flare["start_time"] - merged[-1]["end_time"]) < MERGE_GAP:
            # Merge into previous flare
            prev = merged[-1]
            prev["end_idx"] = flare["end_idx"]
            prev["end_time"] = flare["end_time"]
            prev["duration_sec"] = prev["end_idx"] - prev["start_idx"]
            if flare["peak_counts"] > prev["peak_counts"]:
                prev["peak_idx"] = flare["peak_idx"]
                prev["peak_time"] = flare["peak_time"]
                prev["peak_counts"] = flare["peak_counts"]
                prev["bg_at_peak"] = flare["bg_at_peak"]
                prev["intensity_ratio"] = flare["intensity_ratio"]
            prev["rise_time_sec"] = prev["peak_idx"] - prev["start_idx"]
            prev["decay_time_sec"] = prev["end_idx"] - prev["peak_idx"]
        else:
            merged.append(flare)

    return merged


def classify_flare(peak_counts, bg_counts):
    """
    Classify flare intensity based on peak-to-background ratio.

    Note: Without proper calibration to W/m², we use count ratios
    as a proxy for GOES classification. This mapping will need
    refinement with calibrated data.

    Approximate mapping (counts ratio -> GOES class):
      > 50x background -> X-class
      > 20x background -> M-class
      > 5x  background -> C-class
      > 2x  background -> B-class
    """
    ratio = peak_counts / max(bg_counts, 1)
    if ratio > 50:
        return "X"
    elif ratio > 20:
        return "M"
    elif ratio > 5:
        return "C"
    elif ratio > 2:
        return "B"
    else:
        return "A"


# ═══════════════════════════════════════════════════════════════
# PROCESS ALL DATES
# ═══════════════════════════════════════════════════════════════
def process_all_dates():
    """Process all available dates and build the master catalog."""

    all_flares = []
    daily_summaries = []

    folders = sorted([f for f in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, f)) and f.startswith("AL1_")])

    print(f"Found {len(folders)} days of data\n")
    print("=" * 80)

    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        lc_path, gti_path, detector = find_lc_file(folder_path)

        if not lc_path:
            print(f"[!] {folder}: No lightcurve found, skipping")
            continue

        # Load data
        time_raw, counts, tstart, mjd_ref = load_lightcurve(lc_path)
        date_str = tstart.datetime.strftime("%Y-%m-%d")

        # Handle NaNs
        nan_mask = np.isnan(counts)
        counts_clean = np.nan_to_num(counts, nan=0)

        # Compute background
        print(f"Processing {date_str} ({detector})...", end=" ")
        bg = compute_background_fast(counts_clean)

        # Detect flares
        flares = detect_flares(time_raw, counts_clean, bg)

        # Classify each flare
        for i, flare in enumerate(flares):
            flare_class = classify_flare(flare["peak_counts"], flare["bg_at_peak"])
            flare["flare_class"] = flare_class
            flare["date"] = date_str
            flare["detector"] = detector
            flare["flare_id"] = f"SLX_{date_str.replace('-', '')}_{i+1:03d}"

            # Convert times to UTC strings
            peak_utc = Time(mjd_ref + flare["peak_time"] / 86400.0, format="mjd")
            start_utc = Time(mjd_ref + flare["start_time"] / 86400.0, format="mjd")
            end_utc = Time(mjd_ref + flare["end_time"] / 86400.0, format="mjd")
            flare["peak_utc"] = peak_utc.iso
            flare["start_utc"] = start_utc.iso
            flare["end_utc"] = end_utc.iso

        all_flares.extend(flares)

        # Summary
        n_flares = len(flares)
        classes = [f["flare_class"] for f in flares]
        class_summary = {c: classes.count(c) for c in sorted(set(classes))} if classes else {}
        print(f"-> {n_flares} flare(s) detected: {class_summary}")

        daily_summaries.append({
            "date": date_str,
            "detector": detector,
            "n_flares": n_flares,
            "classes": class_summary,
            "max_counts": float(counts_clean.max()),
            "mean_counts": float(counts_clean.mean()),
        })

        # ── Plot this day with flare markers ──
        plot_day_with_flares(time_raw, counts_clean, bg, flares, date_str, detector, mjd_ref)

    print("=" * 80)
    print(f"\nTotal flares detected: {len(all_flares)}")

    return all_flares, daily_summaries


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def plot_day_with_flares(time_raw, counts, bg, flares, date_str, detector, mjd_ref):
    """Plot a single day's lightcurve with detected flare markers."""

    # Convert to hours from start for x-axis
    t0 = time_raw[0]
    hours = (time_raw - t0) / 3600.0

    # Smooth for cleaner plot
    counts_smooth = uniform_filter1d(counts, size=60)  # 1-min smooth

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot lightcurve
    ax.semilogy(hours, np.where(counts_smooth > 0, counts_smooth, 0.1),
                linewidth=0.6, color="#1565C0", alpha=0.9, label="Counts (1-min smooth)")

    # Plot background
    ax.semilogy(hours, np.where(bg > 0, bg, 0.1),
                linewidth=1, color="gray", alpha=0.6, linestyle="--", label="Background")

    # Plot threshold
    ax.semilogy(hours, bg * RISE_THRESHOLD,
                linewidth=0.8, color="red", alpha=0.3, linestyle=":", label=f"{RISE_THRESHOLD}× threshold")

    # Mark flares
    colors = {"A": "gray", "B": "#4CAF50", "C": "#FFC107", "M": "#FF9800", "X": "#F44336"}
    for flare in flares:
        start_h = (flare["start_time"] - t0) / 3600
        peak_h = (flare["peak_time"] - t0) / 3600
        end_h = (flare["end_time"] - t0) / 3600
        fc = flare["flare_class"]
        color = colors.get(fc, "purple")

        # Shade the flare region
        ax.axvspan(start_h, end_h, alpha=0.15, color=color)

        # Mark start, peak, end
        ax.plot(start_h, counts[flare["start_idx"]], "g^", markersize=8)
        ax.plot(peak_h, flare["peak_counts"], "k*", markersize=12)
        ax.plot(end_h, counts[min(flare["end_idx"], len(counts)-1)], "rv", markersize=8)

        # Label
        peak_utc_short = flare["peak_utc"].split(" ")[1][:8]
        ax.annotate(f'{fc}-class\n{peak_utc_short}\n{flare["peak_counts"]:.0f} cts',
                    xy=(peak_h, flare["peak_counts"]),
                    xytext=(peak_h + 0.3, flare["peak_counts"] * 1.5),
                    fontsize=7, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_xlabel("Hours (UTC)", fontsize=11)
    ax.set_ylabel("Counts/s", fontsize=11)
    ax.set_title(f"SoLEXS {detector} - {date_str} | {len(flares)} flare(s) detected",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3, which="both")

    # Add hour labels
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

    plt.tight_layout()
    out_name = f"flares_{date_str}.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_name}")


# ═══════════════════════════════════════════════════════════════
# MASTER CATALOG
# ═══════════════════════════════════════════════════════════════
def save_catalog(all_flares, daily_summaries):
    """Save the master flare catalog as CSV and JSON."""

    # CSV catalog
    if all_flares:
        catalog_cols = [
            "flare_id", "date", "detector", "flare_class",
            "start_utc", "peak_utc", "end_utc",
            "peak_counts", "mean_counts", "bg_at_peak",
            "duration_sec", "rise_time_sec", "decay_time_sec",
            "intensity_ratio"
        ]
        df = pd.DataFrame(all_flares)[catalog_cols]
        df.to_csv("master_flare_catalog.csv", index=False)
        print(f"\n[CATALOG] Master Catalog saved: master_flare_catalog.csv ({len(df)} flares)")

        # Print the catalog
        print("\n" + "=" * 100)
        print("MASTER FLARE CATALOG - SoLEXS Nowcasting")
        print("=" * 100)
        print(df.to_string(index=False))

        # Summary by class
        print("\n--- Summary by Class ---")
        for cls in ["X", "M", "C", "B", "A"]:
            n = (df["flare_class"] == cls).sum()
            if n > 0:
                print(f"  {cls}-class: {n} flare(s)")
    else:
        print("\n[WARNING] No flares detected in any data!")

    # JSON for dashboard use
    output = {
        "catalog": all_flares,
        "daily_summaries": daily_summaries,
        "config": {
            "smoothing_window": SMOOTHING_WINDOW,
            "bg_window": BG_WINDOW,
            "rise_threshold": RISE_THRESHOLD,
            "min_flare_duration": MIN_FLARE_DURATION,
            "min_peak_counts": MIN_PEAK_COUNTS,
        }
    }
    with open("master_flare_catalog.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("[CATALOG] JSON catalog saved: master_flare_catalog.json")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[*] Solar Flare Detection - SoLEXS Nowcasting Pipeline")
    print("=" * 80)
    all_flares, daily_summaries = process_all_dates()
    save_catalog(all_flares, daily_summaries)
    print("\n[OK] Done! Check the flares_*.png plots and master_flare_catalog.csv")
