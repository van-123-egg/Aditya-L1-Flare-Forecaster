"""
Prepare Training Data for Combined SoLEXS + HEL1OS Flare Forecasting (v4)
==========================================================================
This script runs LOCALLY on your machine. It reads all FITS files from both
instruments, aligns them temporally, extracts features, and saves everything
into a compact .npz file that you upload to Google Colab for training.

Output: training_data_v4.npz (~50-100 MB)

Usage:
    python prepare_training_data.py
"""
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import uniform_filter1d
import os
import sys
import io
import glob
import json
import warnings
import time as timer

warnings.filterwarnings("ignore")

# ==================================================================
# CONFIGURATION
# ==================================================================
DATASET_DIR = "dataset"
HEL1OS_DIR = "hel1os"
COMBINED_CATALOG = "master_combined_catalog.json"
SOLEXS_CATALOG = "master_flare_catalog.json"
OUTPUT_FILE = "training_data_v4.npz"

HISTORY_WINDOW = 3600     # 60-min sliding window (seconds)
STEP_SIZE = 60            # sample every 60s
HORIZON_SEC = 900         # predict 15 min ahead of flare PEAK

# SoLEXS energy bands (channel ranges in 340-channel spectrum)
ENERGY_BANDS = {
    "soft":   (10, 25),
    "medium": (25, 50),
    "hard":   (50, 100),
    "vhard":  (100, 200),
}

# Time-series channels saved for LSTM
TS_CHANNELS = [
    "slx_total", "slx_soft", "slx_medium", "slx_hard", "slx_vhard",
    "hel_cdte", "hel_czt",
]

# Downsampling factor for LSTM time series (1s -> 10s bins)
TS_DOWNSAMPLE = 10
TS_WINDOW_STEPS = HISTORY_WINDOW // TS_DOWNSAMPLE  # 360 steps


# ==================================================================
# SoLEXS DATA LOADING (from forecast_flares_v3 logic)
# ==================================================================
def find_solexs_files(folder_path):
    """Find .lc and .pi files in a SoLEXS day folder."""
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
            pi_files = glob.glob(os.path.join(sdd_dir, "*.pi.gz")) + \
                       glob.glob(os.path.join(sdd_dir, "*.pi"))
            if lc_files:
                return lc_files[0], pi_files[0] if pi_files else None, det
    return None, None, None


def load_solexs_day(folder_path):
    """Load SoLEXS lightcurve + energy-band data for one day."""
    lc_path, pi_path, det = find_solexs_files(folder_path)
    if not lc_path:
        return None

    hdul = fits.open(lc_path)
    time_raw = hdul[1].data["TIME"].copy()
    counts_total = np.nan_to_num(hdul[1].data["COUNTS"].copy(), nan=0.0)
    header = hdul[1].header
    mjd_ref = header["MJDREFI"] + header["MJDREFF"]
    tstart = Time(mjd_ref + header["TSTART"] / 86400.0, format="mjd")
    date_str = tstart.datetime.strftime("%Y-%m-%d")
    hdul.close()

    band_data = {}
    if pi_path:
        try:
            hdul_pi = fits.open(pi_path)
            spec_counts = hdul_pi[1].data["COUNTS"]
            for band_name, (ch_lo, ch_hi) in ENERGY_BANDS.items():
                band_lc = np.nansum(spec_counts[:, ch_lo:ch_hi], axis=1)
                band_data[band_name] = band_lc.astype(np.float64)
            hdul_pi.close()
        except Exception as e:
            print(f"    [!] Spectrum error: {e}")

    return {
        "date": date_str,
        "times": time_raw,
        "counts": counts_total,
        "bands": band_data,
        "has_spectrum": len(band_data) > 0,
    }


# ==================================================================
# HEL1OS DATA LOADING (import from existing module)
# ==================================================================
def load_hel1os_day_quiet(day_folder):
    """Load HEL1OS data, suppressing stdout."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from detect_flares_hel1os import load_hel1os_day
        data = load_hel1os_day(day_folder)
    finally:
        sys.stdout = old_stdout
    return data


# ==================================================================
# FEATURE ENGINEERING (v4 — SoLEXS + HEL1OS combined)
# ==================================================================
def engineer_features_v4(times, counts, bands, has_spectrum,
                         hel_cdte, hel_czt):
    """
    Extract tabular features for XGBoost.

    Extends v3 with HEL1OS features:
    - HEL1OS CdTe/CZT: statistics, derivatives, trends
    - Cross-instrument ratios and hardness indices
    - HEL1OS impulsiveness (derivative variance)

    Also returns raw downsampled time-series windows for LSTM.
    """
    n = len(counts)

    # Pre-compute smoothed SoLEXS
    smooth_30s = uniform_filter1d(counts, size=30)
    smooth_60s = uniform_filter1d(counts, size=60)
    smooth_300s = uniform_filter1d(counts, size=300)
    deriv_slx = np.gradient(smooth_30s)
    log_counts = np.log1p(counts)

    band_smooth = {}
    band_deriv = {}
    if has_spectrum:
        for bname, blc in bands.items():
            bs = uniform_filter1d(blc, size=60)
            band_smooth[bname] = bs
            band_deriv[bname] = np.gradient(bs)

    # Pre-compute smoothed HEL1OS
    hel_smooth_60 = uniform_filter1d(hel_cdte, size=60)
    hel_smooth_300 = uniform_filter1d(hel_cdte, size=300)
    hel_deriv = np.gradient(hel_smooth_60)

    czt_smooth_60 = uniform_filter1d(hel_czt, size=60)
    czt_deriv = np.gradient(czt_smooth_60)

    features_list = []
    sample_indices = []
    ts_windows = []  # for LSTM

    for i in range(HISTORY_WINDOW, n, STEP_SIZE):
        win = counts[i - HISTORY_WINDOW:i]
        win_deriv = deriv_slx[i - HISTORY_WINDOW:i]
        win_log = log_counts[i - HISTORY_WINDOW:i]
        recent_5min = counts[max(0, i - 300):i]
        recent_10min = counts[max(0, i - 600):i]
        earlier = counts[i - HISTORY_WINDOW:max(0, i - 600)]

        feat = {}

        # ─── SoLEXS total counts features (same as v3) ───
        feat["mean"] = np.mean(win)
        feat["std"] = np.std(win)
        feat["max"] = np.max(win)
        feat["median"] = np.median(win)
        m, s = feat["mean"], feat["std"]
        feat["skewness"] = np.mean(((win - m) / max(s, 0.01)) ** 3)
        feat["kurtosis"] = np.mean(((win - m) / max(s, 0.01)) ** 4) - 3
        feat["iqr"] = np.percentile(win, 75) - np.percentile(win, 25)
        feat["p90"] = np.percentile(win, 90)
        feat["p99"] = np.percentile(win, 99)
        feat["log_mean"] = np.mean(win_log)
        feat["log_std"] = np.std(win_log)
        feat["current"] = smooth_30s[i - 1]
        feat["current_raw"] = counts[i - 1]
        feat["ratio_60_300"] = smooth_60s[i - 1] / max(smooth_300s[i - 1], 0.1)

        # Derivatives
        feat["deriv_mean"] = np.mean(win_deriv)
        feat["deriv_max"] = np.max(win_deriv)
        feat["deriv_current"] = deriv_slx[i - 1]
        recent_deriv = deriv_slx[max(0, i - 300):i]
        feat["deriv_recent_mean"] = np.mean(recent_deriv)
        feat["deriv_recent_max"] = np.max(recent_deriv)

        # Trend
        feat["recent_vs_earlier"] = np.mean(recent_5min) / max(np.mean(earlier), 0.1) if len(earlier) > 0 else 1.0
        feat["recent10_vs_earlier"] = np.mean(recent_10min) / max(np.mean(earlier), 0.1) if len(earlier) > 0 else 1.0
        x = np.arange(len(win))
        coeffs = np.polyfit(x, uniform_filter1d(win, size=30), 1)
        feat["trend_slope"] = coeffs[0]

        # Background
        bg = np.percentile(win, 30)
        feat["background"] = bg
        feat["current_over_bg"] = smooth_30s[i - 1] / max(bg, 0.1)

        # Monotonic rise
        n_seg = 6
        seg_len = HISTORY_WINDOW // n_seg
        seg_means = [np.mean(win[j * seg_len:(j + 1) * seg_len]) for j in range(n_seg)]
        rising = sum(1 for j in range(1, n_seg) if seg_means[j] > seg_means[j - 1])
        feat["monotonic_rise"] = rising / (n_seg - 1)
        feat["seg_ratio"] = seg_means[-1] / max(seg_means[0], 0.1)
        feat["cv"] = np.std(win) / max(np.mean(win), 0.1)

        # ─── SoLEXS energy band features (same as v3) ───
        if has_spectrum:
            for bname in ENERGY_BANDS:
                blc = bands[bname]
                bwin = blc[i - HISTORY_WINDOW:i]
                bs = band_smooth[bname]
                bd = band_deriv[bname]

                feat[f"{bname}_mean"] = np.mean(bwin)
                feat[f"{bname}_max"] = np.max(bwin)
                feat[f"{bname}_current"] = bs[i - 1]
                feat[f"{bname}_std"] = np.std(bwin)

                bd_win = bd[i - HISTORY_WINDOW:i]
                feat[f"{bname}_deriv_mean"] = np.mean(bd_win)
                feat[f"{bname}_deriv_max"] = np.max(bd_win)
                feat[f"{bname}_deriv_current"] = bd[i - 1]

                b_recent = blc[max(0, i - 300):i]
                b_earlier = blc[i - HISTORY_WINDOW:max(0, i - 600)]
                feat[f"{bname}_recent_trend"] = np.mean(b_recent) / max(np.mean(b_earlier), 0.01) if len(b_earlier) > 0 else 1.0

                b_bg = np.percentile(bwin, 30)
                feat[f"{bname}_over_bg"] = bs[i - 1] / max(b_bg, 0.01)

                b_segs = [np.mean(bwin[j * seg_len:(j + 1) * seg_len]) for j in range(n_seg)]
                b_rising = sum(1 for j in range(1, n_seg) if b_segs[j] > b_segs[j - 1])
                feat[f"{bname}_mono_rise"] = b_rising / (n_seg - 1)

            # Hardness ratios (SoLEXS internal)
            soft_val = max(band_smooth["soft"][i - 1], 0.01)
            med_val = max(band_smooth["medium"][i - 1], 0.01)
            hard_val = band_smooth["hard"][i - 1]
            vhard_val = band_smooth["vhard"][i - 1]

            feat["hardness_hard_soft"] = hard_val / soft_val
            feat["hardness_vhard_soft"] = vhard_val / soft_val
            feat["hardness_hard_med"] = hard_val / med_val
            feat["hardness_vhard_med"] = vhard_val / med_val

            hr_window = bands["hard"][max(0, i - 300):i] / np.maximum(bands["soft"][max(0, i - 300):i], 0.01)
            hr_earlier = bands["hard"][i - HISTORY_WINDOW:max(0, i - 600)] / np.maximum(bands["soft"][i - HISTORY_WINDOW:max(0, i - 600)], 0.01)
            feat["hardness_trend"] = np.mean(hr_window) / max(np.mean(hr_earlier), 0.001) if len(hr_earlier) > 0 else 1.0

            hr_full = bands["hard"][i - HISTORY_WINDOW:i] / np.maximum(bands["soft"][i - HISTORY_WINDOW:i], 0.01)
            hr_smooth = uniform_filter1d(hr_full, size=60)
            feat["hardness_deriv"] = np.gradient(hr_smooth)[-1]
            feat["hardness_deriv_max"] = np.max(np.gradient(hr_smooth)[-300:])

            feat["hard_active"] = float(np.mean(bands["hard"][max(0, i - 300):i]) > 1.0)
            feat["vhard_active"] = float(np.mean(bands["vhard"][max(0, i - 60):i]) > 0.5)

        # ─── NEW: HEL1OS CdTe features ───
        hwin = hel_cdte[i - HISTORY_WINDOW:i]
        feat["hel_mean"] = np.mean(hwin)
        feat["hel_std"] = np.std(hwin)
        feat["hel_max"] = np.max(hwin)
        feat["hel_median"] = np.median(hwin)
        feat["hel_current"] = hel_smooth_60[i - 1]
        feat["hel_p90"] = np.percentile(hwin, 90)
        feat["hel_p99"] = np.percentile(hwin, 99)
        feat["hel_log_mean"] = np.mean(np.log1p(hwin))

        hwin_deriv = hel_deriv[i - HISTORY_WINDOW:i]
        feat["hel_deriv_mean"] = np.mean(hwin_deriv)
        feat["hel_deriv_max"] = np.max(hwin_deriv)
        feat["hel_deriv_current"] = hel_deriv[i - 1]
        feat["hel_deriv_recent_mean"] = np.mean(hel_deriv[max(0, i - 300):i])
        feat["hel_deriv_recent_max"] = np.max(hel_deriv[max(0, i - 300):i])

        hel_recent = hel_cdte[max(0, i - 300):i]
        hel_earlier = hel_cdte[i - HISTORY_WINDOW:max(0, i - 600)]
        feat["hel_recent_trend"] = np.mean(hel_recent) / max(np.mean(hel_earlier), 1e-4) if len(hel_earlier) > 0 else 1.0

        hel_bg = np.percentile(hwin, 20)
        feat["hel_background"] = hel_bg
        feat["hel_over_bg"] = hel_smooth_60[i - 1] / max(hel_bg, 1e-4)

        hel_segs = [np.mean(hwin[j * seg_len:(j + 1) * seg_len]) for j in range(n_seg)]
        hel_rising = sum(1 for j in range(1, n_seg) if hel_segs[j] > hel_segs[j - 1])
        feat["hel_mono_rise"] = hel_rising / (n_seg - 1)
        feat["hel_seg_ratio"] = hel_segs[-1] / max(hel_segs[0], 1e-4)
        feat["hel_cv"] = np.std(hwin) / max(np.mean(hwin), 1e-4)

        # Impulsiveness: how bursty the hard X-rays are (high = pre-flare)
        feat["hel_impulsiveness"] = np.std(hwin_deriv) / max(np.mean(np.abs(hwin_deriv)), 1e-4)

        # ─── NEW: HEL1OS CZT features ───
        cwin = hel_czt[i - HISTORY_WINDOW:i]
        feat["czt_mean"] = np.mean(cwin)
        feat["czt_std"] = np.std(cwin)
        feat["czt_max"] = np.max(cwin)
        feat["czt_current"] = czt_smooth_60[i - 1]
        feat["czt_deriv_mean"] = np.mean(czt_deriv[i - HISTORY_WINDOW:i])
        feat["czt_deriv_max"] = np.max(czt_deriv[i - HISTORY_WINDOW:i])
        feat["czt_deriv_current"] = czt_deriv[i - 1]
        czt_recent = hel_czt[max(0, i - 300):i]
        czt_earlier = hel_czt[i - HISTORY_WINDOW:max(0, i - 600)]
        feat["czt_recent_trend"] = np.mean(czt_recent) / max(np.mean(czt_earlier), 1e-2) if len(czt_earlier) > 0 else 1.0

        # ─── NEW: Cross-instrument features ───
        slx_current = max(smooth_30s[i - 1], 0.1)
        hel_current = max(hel_smooth_60[i - 1], 1e-4)

        feat["cross_hel_slx_ratio"] = hel_current / slx_current
        feat["cross_czt_slx_ratio"] = max(czt_smooth_60[i - 1], 1e-4) / slx_current

        # Cross-instrument derivative comparison (hard rising faster = pre-flare)
        slx_deriv_recent = np.mean(deriv_slx[max(0, i - 300):i])
        hel_deriv_recent = np.mean(hel_deriv[max(0, i - 300):i])
        feat["cross_deriv_ratio"] = hel_deriv_recent / max(abs(slx_deriv_recent), 1e-4)

        # HEL1OS-based hardness (using CdTe/SoLEXS soft band)
        if has_spectrum:
            feat["cross_hardness_hel_soft"] = hel_current / max(band_smooth["soft"][i - 1], 0.01)
            feat["cross_hardness_czt_soft"] = max(czt_smooth_60[i - 1], 1e-4) / max(band_smooth["soft"][i - 1], 0.01)

        # ─── Build time-series window for LSTM ───
        ts_win_start = i - HISTORY_WINDOW
        ts_raw = np.column_stack([
            counts[ts_win_start:i],
            bands.get("soft", np.zeros(HISTORY_WINDOW))[ts_win_start:i] if has_spectrum else np.zeros(HISTORY_WINDOW),
            bands.get("medium", np.zeros(HISTORY_WINDOW))[ts_win_start:i] if has_spectrum else np.zeros(HISTORY_WINDOW),
            bands.get("hard", np.zeros(HISTORY_WINDOW))[ts_win_start:i] if has_spectrum else np.zeros(HISTORY_WINDOW),
            bands.get("vhard", np.zeros(HISTORY_WINDOW))[ts_win_start:i] if has_spectrum else np.zeros(HISTORY_WINDOW),
            hel_cdte[ts_win_start:i],
            hel_czt[ts_win_start:i],
        ])  # shape: (3600, 7)

        # Downsample by averaging every TS_DOWNSAMPLE points
        n_steps = ts_raw.shape[0] // TS_DOWNSAMPLE
        ts_downsampled = ts_raw[:n_steps * TS_DOWNSAMPLE].reshape(n_steps, TS_DOWNSAMPLE, -1).mean(axis=1)
        ts_windows.append(ts_downsampled)  # shape: (360, 7)

        features_list.append(feat)
        sample_indices.append(i)

    df = pd.DataFrame(features_list)
    # Fill any NaN/inf with 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    ts_array = np.array(ts_windows, dtype=np.float32)

    return df.values.astype(np.float32), list(df.columns), np.array(sample_indices), ts_array


def create_labels(times, catalog_entries, solexs_cat, horizon_sec=HORIZON_SEC):
    """Create binary labels: 1 if a flare peaks within horizon_sec."""
    labels = np.zeros(len(times), dtype=np.int32)
    for entry in catalog_entries:
        # We need the raw peak_time (seconds from TSTART) to match 'times'.
        # This is stored in the original SoLEXS catalog.
        slx_id = entry.get("slx_flare_id")
        peak_time = None
        
        if slx_id:
            # Find it in solexs_cat
            for slx_entry in solexs_cat:
                if slx_entry["flare_id"] == slx_id:
                    peak_time = slx_entry["peak_time"]
                    break
                    
        if peak_time is None:
            # Fallback for HEL1OS-only flares: approximate from UTC
            hel_utc = entry.get("hel_peak_utc")
            if not hel_utc:
                continue
            from datetime import datetime
            try:
                peak_dt = datetime.fromisoformat(hel_utc)
            except (ValueError, TypeError):
                continue
            
            # This fallback is imperfect because 'times' is from TSTART, 
            # but we'll approximate assuming TSTART is near midnight if we must.
            # (Better to just skip HEL1OS-only flares for training, or use the 
            # SoLEXS time reference). Since we interpolate HEL1OS to SoLEXS times,
            # we really need the SoLEXS TSTART. 
            continue

        mask = (times >= peak_time - horizon_sec) & (times < peak_time)
        labels[mask] = 1

    return labels


# ==================================================================
# MAIN
# ==================================================================
def main():
    t_start = timer.time()
    print("=" * 70)
    print("  PREPARE TRAINING DATA v4 (SoLEXS + HEL1OS Combined)")
    print("=" * 70)

    # 1. Find overlapping days
    solexs_folders = sorted([f for f in os.listdir(DATASET_DIR)
                             if os.path.isdir(os.path.join(DATASET_DIR, f)) and f.startswith("AL1_")])
    hel1os_folders = sorted([f for f in os.listdir(HEL1OS_DIR)
                             if os.path.isdir(os.path.join(HEL1OS_DIR, f)) and f.startswith("HLS_")])

    solexs_by_date = {}
    for f in solexs_folders:
        date_part = f.split("_")[3]  # AL1_SLX_L1_20260602_v1.0 -> 20260602
        solexs_by_date[date_part] = os.path.join(DATASET_DIR, f)

    hel1os_by_date = {}
    for f in hel1os_folders:
        date_part = f.split("_")[1]  # HLS_20260602 -> 20260602
        hel1os_by_date[date_part] = os.path.join(HEL1OS_DIR, f)

    overlap_dates = sorted(set(solexs_by_date.keys()) & set(hel1os_by_date.keys()))
    print(f"\n  SoLEXS days: {len(solexs_by_date)}")
    print(f"  HEL1OS days: {len(hel1os_by_date)}")
    print(f"  Overlapping: {len(overlap_dates)}")

    # 2. Load combined catalog
    with open(COMBINED_CATALOG, "r") as f:
        combined_cat = json.load(f)["catalog"]
    print(f"  Combined catalog: {len(combined_cat)} entries")

    # Also load SoLEXS-only catalog for labeling (it has peak_time in seconds)
    with open(SOLEXS_CATALOG, "r") as f:
        solexs_cat = json.load(f)["catalog"]

    # 3. Process each overlapping day
    print(f"\n{'=' * 70}")
    print("  Processing days...")
    print(f"{'=' * 70}")

    all_X_tabular = []
    all_X_ts = []
    all_y = []
    all_day_idx = []
    all_sample_times = []
    day_dates = []
    feature_names = None

    for day_i, date_str in enumerate(overlap_dates):
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        print(f"\n  [{day_i + 1}/{len(overlap_dates)}] {date_fmt}")

        # Load SoLEXS
        print(f"    Loading SoLEXS...", end=" ")
        slx_data = load_solexs_day(solexs_by_date[date_str])
        if slx_data is None:
            print("FAILED - skipping")
            continue
        if not slx_data["has_spectrum"]:
            print("NO SPECTRUM - skipping")
            continue
        print(f"OK ({len(slx_data['counts'])} pts)")

        # Load HEL1OS
        print(f"    Loading HEL1OS...", end=" ")
        hel_data = load_hel1os_day_quiet(hel1os_by_date[date_str])
        if hel_data is None:
            print("FAILED - skipping")
            continue
        print(f"OK ({len(hel_data['total_ctr'])} pts)")

        # Align HEL1OS to SoLEXS time grid by interpolation
        slx_times = slx_data["times"]
        hel_times = hel_data["times"]
        n_slx = len(slx_times)

        hel_cdte_aligned = np.interp(slx_times, hel_times, hel_data["total_ctr"],
                                     left=0.0, right=0.0)
        if hel_data["czt_ctr"] is not None:
            hel_czt_aligned = np.interp(slx_times, hel_times, hel_data["czt_ctr"],
                                        left=0.0, right=0.0)
        else:
            hel_czt_aligned = np.zeros(n_slx)

        # Clean up NaN/negatives
        hel_cdte_aligned = np.nan_to_num(hel_cdte_aligned, nan=0.0)
        hel_czt_aligned = np.nan_to_num(hel_czt_aligned, nan=0.0)
        hel_cdte_aligned = np.clip(hel_cdte_aligned, 0, None)
        hel_czt_aligned = np.clip(hel_czt_aligned, 0, None)

        # Extract features
        print(f"    Extracting features...", end=" ")
        X_tab, feat_names, sample_idx, X_ts = engineer_features_v4(
            slx_times, slx_data["counts"], slx_data["bands"],
            slx_data["has_spectrum"], hel_cdte_aligned, hel_czt_aligned
        )
        print(f"{X_tab.shape[0]} samples, {X_tab.shape[1]} features, TS: {X_ts.shape}")

        if feature_names is None:
            feature_names = feat_names
        else:
            assert feature_names == feat_names, "Feature names mismatch between days!"

        # Create labels from combined catalog
        day_catalog = [e for e in combined_cat if e["date"] == date_fmt]
        y = create_labels(slx_times, day_catalog, solexs_cat, HORIZON_SEC)[sample_idx]

        n_pos = y.sum()
        print(f"    Labels: {n_pos} positive ({100 * n_pos / len(y):.1f}%), "
              f"{len(day_catalog)} catalog entries")

        all_X_tabular.append(X_tab)
        all_X_ts.append(X_ts)
        all_y.append(y)
        all_day_idx.append(np.full(len(y), day_i, dtype=np.int32))
        all_sample_times.append(slx_times[sample_idx])
        day_dates.append(date_fmt)

    # 4. Combine all days
    print(f"\n{'=' * 70}")
    print("  Combining all days...")
    print(f"{'=' * 70}")

    X_tabular = np.vstack(all_X_tabular)
    X_ts = np.vstack(all_X_ts)
    y = np.concatenate(all_y)
    day_idx = np.concatenate(all_day_idx)
    sample_times = np.concatenate(all_sample_times)

    print(f"\n  Total samples:       {len(y)}")
    print(f"  Positive samples:    {y.sum()} ({100 * y.sum() / len(y):.1f}%)")
    print(f"  Tabular shape:       {X_tabular.shape}")
    print(f"  Time-series shape:   {X_ts.shape}")
    print(f"  Days:                {len(day_dates)}")
    print(f"  Features:            {len(feature_names)}")

    # 5. Save to .npz
    print(f"\n  Saving to {OUTPUT_FILE}...")

    config = {
        "history_window": HISTORY_WINDOW,
        "step_size": STEP_SIZE,
        "horizon_sec": HORIZON_SEC,
        "energy_bands": ENERGY_BANDS,
        "ts_channels": TS_CHANNELS,
        "ts_downsample": TS_DOWNSAMPLE,
        "ts_window_steps": TS_WINDOW_STEPS,
    }

    np.savez_compressed(
        OUTPUT_FILE,
        X_tabular=X_tabular,
        X_ts=X_ts,
        y=y,
        day_idx=day_idx,
        sample_times=sample_times,
        feature_names=np.array(feature_names),
        day_dates=np.array(day_dates),
        channel_names=np.array(TS_CHANNELS),
        config_json=json.dumps(config),
        catalog_json=json.dumps(combined_cat),
    )

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    elapsed = timer.time() - t_start
    print(f"\n  Saved: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"  Elapsed: {elapsed:.0f} seconds")

    print(f"\n{'=' * 70}")
    print("  [OK] Training data preparation complete!")
    print(f"{'=' * 70}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Upload '{OUTPUT_FILE}' to Google Colab")
    print(f"  2. Upload 'train_combined_model_v4.py' to Google Colab")
    print(f"  3. Run: !python train_combined_model_v4.py")
    print(f"  4. Download the trained model files back to this folder")


if __name__ == "__main__":
    main()
