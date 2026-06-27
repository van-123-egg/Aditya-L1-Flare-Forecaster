"""
Solar Flare Detection Algorithm (Nowcasting) - HEL1OS Hard X-ray
=================================================================
Detects and classifies solar flares from HEL1OS CdTe/CZT lightcurve data.
Processes all available dates and generates a HEL1OS Flare Catalog.

HEL1OS provides hard X-ray observations (5-160 keV) from two detector types:
  - CdTe (Cadmium Telluride): 5-90 keV, two detector units (cdte1, cdte2)
  - CZT (Cadmium Zinc Telluride): 18-160 keV, two units (czt1, czt2)

Each day has 2 half-day observation windows (AM + PM) stored as separate
zip files. This script works on already-extracted data.

Key differences from SoLEXS pipeline:
  - Time column: MJD (Modified Julian Day) instead of TIME (seconds from TSTART)
  - Rate column: CTR (counts/sec) instead of COUNTS (raw counts)
  - Much lower count rates: CdTe total mean ~0.19 cts/s vs SoLEXS ~10-50 cts/s
  - Multiple energy bands in separate HDU extensions
  - Two half-day files per day need to be combined
  - Two detector units per type need to be summed
"""
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os
import glob
import json
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
HEL1OS_DIR = "hel1os"
OUTPUT_DIR = "hel1os_plots"

# Detection parameters — adapted for hard X-ray (lower count rates)
SMOOTHING_WINDOW = 60       # seconds — heavier smoothing for sparser data
BG_WINDOW = 600             # 10-minute window for background estimation
BG_PERCENTILE = 20          # lower percentile since HEL1OS counts are sparser
RISE_THRESHOLD = 5.0        # higher multiplier needed (lower background)
MIN_FLARE_DURATION = 20     # hard X-ray flares are shorter/more impulsive
MIN_PEAK_COUNTS = 1.0       # CTR units (cts/s), not raw counts like SoLEXS
MERGE_GAP = 120             # merge detections within 2 minutes

# Energy band definitions for reference
CDTE_BANDS = {
    1: "5.00-20.00 keV",
    2: "20.00-30.00 keV",
    3: "30.00-40.00 keV",
    4: "40.00-60.00 keV",
    5: "1.80-90.00 keV (total)",
}
CZT_BANDS = {
    1: "20.00-40.00 keV",
    2: "40.00-60.00 keV",
    3: "60.00-80.00 keV",
    4: "80.00-150.00 keV",
    5: "18.00-160.00 keV (total)",
}


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def _find_lightcurve_files(day_folder):
    """
    Find all lightcurve FITS files in an HEL1OS day folder.

    The extracted directory structure is:
      HLS_YYYYMMDD/
        HLS_YYYYMMDD_HHMMSS_*_lev1_V*/
          2026/06/DD/
            HLS_YYYYMMDD_*_lev1_V*/
              cdte/lightcurve_cdte1.fits, lightcurve_cdte2.fits
              czt/lightcurve_czt1.fits,  lightcurve_czt2.fits

    Returns a list of dicts, each containing:
      - version: 'V111' or 'V211'
      - half: 'AM' or 'PM' (inferred from start time in folder name)
      - cdte_files: list of CdTe lightcurve file paths
      - czt_files: list of CZT lightcurve file paths
      - start_time_str: HHMMSS from the folder name
    """
    results = []

    # Find all extracted subfolders (not zip files)
    try:
        entries = os.listdir(day_folder)
    except FileNotFoundError:
        print(f"  [!] Day folder not found: {day_folder}")
        return results

    # Match the top-level extracted directories: HLS_YYYYMMDD_HHMMSS_*_lev1_V*
    pattern = re.compile(r'^(HLS_(\d{8})_(\d{6})_\d+sec_lev1_(V\d+))$')
    extraction_dirs = []

    for entry in entries:
        m = pattern.match(entry)
        if m and os.path.isdir(os.path.join(day_folder, entry)):
            extraction_dirs.append({
                "dirname": m.group(1),
                "date_str": m.group(2),
                "time_str": m.group(3),
                "version": m.group(4),
                "path": os.path.join(day_folder, entry),
            })

    if not extraction_dirs:
        print(f"  [!] No extracted folders found in {day_folder}")
        print(f"      (Zip files must be extracted before running this script)")
        return results

    # Group by approximate half-day (AM: start before 12:00, PM: start at/after 12:00)
    for ed in extraction_dirs:
        hour = int(ed["time_str"][:2])
        half = "AM" if hour < 12 else "PM"

        # Navigate to the nested lightcurve directory
        # Pattern: <extraction_dir>/2026/06/DD/<inner_dir>/cdte/ and /czt/
        inner_lc = _find_nested_lightcurves(ed["path"], ed["dirname"])
        if inner_lc is None:
            continue

        results.append({
            "version": ed["version"],
            "half": half,
            "start_time_str": ed["time_str"],
            "cdte_files": inner_lc["cdte"],
            "czt_files": inner_lc["czt"],
        })

    return results


def _find_nested_lightcurves(extraction_path, dirname):
    """
    Navigate the nested directory tree to find lightcurve files.
    Structure: <extraction_path>/YYYY/MM/DD/<inner_dirname>/cdte/ and /czt/
    """
    cdte_files = []
    czt_files = []

    # Use glob to find the lightcurve files deep in the tree
    # Pattern: **/cdte/lightcurve_cdte*.fits
    cdte_pattern = os.path.join(extraction_path, "**", "cdte", "lightcurve_cdte*.fits")
    czt_pattern = os.path.join(extraction_path, "**", "czt", "lightcurve_czt*.fits")

    cdte_files = sorted(glob.glob(cdte_pattern, recursive=True))
    czt_files = sorted(glob.glob(czt_pattern, recursive=True))

    if not cdte_files and not czt_files:
        return None

    return {"cdte": cdte_files, "czt": czt_files}


def _select_best_version(file_groups):
    """
    If both V111 and V211 exist for the same half-day, prefer V211 (newer).
    Returns filtered list with one entry per half-day.
    """
    # Group by half-day
    by_half = {}
    for fg in file_groups:
        half = fg["half"]
        if half not in by_half:
            by_half[half] = []
        by_half[half].append(fg)

    selected = []
    for half, groups in sorted(by_half.items()):
        if len(groups) == 1:
            selected.append(groups[0])
        else:
            # Prefer V211 over V111
            v211 = [g for g in groups if g["version"] == "V211"]
            if v211:
                selected.append(v211[0])
            else:
                # Take the highest version number
                groups.sort(key=lambda g: g["version"], reverse=True)
                selected.append(groups[0])

    return selected


def _load_lightcurve_fits(filepath, total_hdu_idx=5):
    """
    Load a single HEL1OS lightcurve FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    total_hdu_idx : int
        HDU index for the total-band lightcurve (default: 5).

    Returns
    -------
    dict with keys:
      - 'mjd': array of MJD timestamps
      - 'ctr': array of count rates (cts/s)
      - 'stat_err': array of statistical errors
      - 'bands': dict of {hdu_idx: {'mjd', 'ctr', 'stat_err', 'band_name'}}
    """
    try:
        hdul = fits.open(filepath)
    except Exception as e:
        print(f"  [!] Cannot open {filepath}: {e}")
        return None

    result = {"bands": {}}

    # Load total band from the specified HDU
    try:
        total_data = hdul[total_hdu_idx].data
        total_header = hdul[total_hdu_idx].header
        result["mjd"] = total_data["MJD"].astype(np.float64).copy()
        result["ctr"] = total_data["CTR"].astype(np.float64).copy()
        result["stat_err"] = total_data["STAT_ERR"].astype(np.float64).copy()
        band_name = total_header.get("EXTNAME", f"HDU{total_hdu_idx}")
        result["total_band_name"] = band_name
    except Exception as e:
        print(f"  [!] Cannot read total band (HDU[{total_hdu_idx}]) from {filepath}: {e}")
        hdul.close()
        return None

    # Load individual energy bands (HDUs 1-4 typically)
    for hdu_idx in range(1, len(hdul)):
        if hdu_idx == total_hdu_idx:
            continue  # already loaded
        try:
            data = hdul[hdu_idx].data
            header = hdul[hdu_idx].header
            if data is not None and "MJD" in data.columns.names and "CTR" in data.columns.names:
                band_name = header.get("EXTNAME", f"HDU{hdu_idx}")
                result["bands"][hdu_idx] = {
                    "mjd": data["MJD"].astype(np.float64).copy(),
                    "ctr": data["CTR"].astype(np.float64).copy(),
                    "stat_err": data["STAT_ERR"].astype(np.float64).copy(),
                    "band_name": band_name,
                }
        except Exception:
            pass  # skip non-data HDUs (e.g., primary HDU)

    hdul.close()
    return result


def _combine_detector_pair(data1, data2):
    """
    Combine two detector units (e.g., cdte1 + cdte2) by time-aligning on MJD
    and summing the count rates.

    Uses the intersection of MJD timestamps (rounded to nearest millisecond
    to handle floating-point differences).
    """
    if data1 is None and data2 is None:
        return None
    if data1 is None:
        return data2
    if data2 is None:
        return data1

    # Round MJD to ~1ms precision for matching (1ms ≈ 1.16e-8 days)
    precision = 8  # decimal places in MJD
    mjd1_r = np.round(data1["mjd"], precision)
    mjd2_r = np.round(data2["mjd"], precision)

    # Find common timestamps using set intersection
    common_mjd, idx1, idx2 = np.intersect1d(mjd1_r, mjd2_r, return_indices=True)

    if len(common_mjd) == 0:
        # No overlap — concatenate instead (different time segments)
        combined_mjd = np.concatenate([data1["mjd"], data2["mjd"]])
        combined_ctr = np.concatenate([data1["ctr"], data2["ctr"]])
        combined_err = np.concatenate([data1["stat_err"], data2["stat_err"]])
        sort_idx = np.argsort(combined_mjd)
        return {
            "mjd": combined_mjd[sort_idx],
            "ctr": combined_ctr[sort_idx],
            "stat_err": combined_err[sort_idx],
        }

    # Sum count rates at matching timestamps
    combined_ctr = data1["ctr"][idx1] + data2["ctr"][idx2]
    # Error propagation: sqrt(err1² + err2²)
    combined_err = np.sqrt(data1["stat_err"][idx1]**2 + data2["stat_err"][idx2]**2)

    return {
        "mjd": data1["mjd"][idx1],
        "ctr": combined_ctr,
        "stat_err": combined_err,
    }


def _combine_halves(am_data, pm_data):
    """
    Combine AM and PM half-day data into a full day by concatenation.
    """
    if am_data is None and pm_data is None:
        return None
    if am_data is None:
        return pm_data
    if pm_data is None:
        return am_data

    combined_mjd = np.concatenate([am_data["mjd"], pm_data["mjd"]])
    combined_ctr = np.concatenate([am_data["ctr"], pm_data["ctr"]])
    combined_err = np.concatenate([am_data["stat_err"], pm_data["stat_err"]])

    # Sort by time (should already be in order, but be safe)
    sort_idx = np.argsort(combined_mjd)
    return {
        "mjd": combined_mjd[sort_idx],
        "ctr": combined_ctr[sort_idx],
        "stat_err": combined_err[sort_idx],
    }


def _combine_bands_for_half(fg, detector_type="cdte"):
    """
    Load and combine both detector units for all energy bands for one half-day.

    Returns dict: {band_idx: combined_data_dict, 'total': combined_total}
    """
    if detector_type == "cdte":
        files = fg["cdte_files"]
    else:
        files = fg["czt_files"]

    if not files:
        return None

    # Load each detector unit
    loaded = []
    for f in sorted(files):
        data = _load_lightcurve_fits(f, total_hdu_idx=5)
        if data is not None:
            loaded.append(data)

    if not loaded:
        return None

    # Combine the two detector units for the total band
    if len(loaded) >= 2:
        total_combined = _combine_detector_pair(loaded[0], loaded[1])
    else:
        total_combined = {
            "mjd": loaded[0]["mjd"],
            "ctr": loaded[0]["ctr"],
            "stat_err": loaded[0]["stat_err"],
        }

    # Combine individual bands
    band_combined = {}
    all_band_indices = set()
    for ld in loaded:
        all_band_indices.update(ld["bands"].keys())

    for bidx in sorted(all_band_indices):
        band_data_list = []
        for ld in loaded:
            if bidx in ld["bands"]:
                b = ld["bands"][bidx]
                band_data_list.append({
                    "mjd": b["mjd"],
                    "ctr": b["ctr"],
                    "stat_err": b["stat_err"],
                })
        if len(band_data_list) >= 2:
            band_combined[bidx] = _combine_detector_pair(band_data_list[0], band_data_list[1])
        elif len(band_data_list) == 1:
            band_combined[bidx] = band_data_list[0]

        # Attach band name from the first loaded file
        for ld in loaded:
            if bidx in ld["bands"]:
                band_combined[bidx]["band_name"] = ld["bands"][bidx]["band_name"]
                break

    return {"total": total_combined, "bands": band_combined}


def load_hel1os_day(day_folder):
    """
    Load a full day of HEL1OS data.

    Parameters
    ----------
    day_folder : str
        Path to the day folder, e.g., 'hel1os/HLS_20260601'.

    Returns
    -------
    dict with keys:
      - 'times': seconds from day start (float64 array)
      - 'total_ctr': combined CdTe total-band count rate (cts/s)
      - 'czt_ctr': combined CZT total-band count rate (cts/s)
      - 'total_err': combined CdTe statistical error
      - 'czt_err': combined CZT statistical error
      - 'band_data': dict of {detector_type: {band_idx: {mjd, ctr, stat_err}}}
      - 'date_str': date string like '2026-06-01'
      - 'mjd_ref': MJD of day start (midnight)
      - 'mjd': raw MJD array for CdTe total
    Returns None if no data can be loaded.
    """
    print(f"\n  Loading {os.path.basename(day_folder)}...")

    # Step 1: Find all lightcurve file groups
    file_groups = _find_lightcurve_files(day_folder)
    if not file_groups:
        return None

    # Step 2: Select best version (prefer V211)
    selected = _select_best_version(file_groups)
    print(f"    Found {len(selected)} half-day segment(s): "
          f"{[s['half'] + ' (' + s['version'] + ')' for s in selected]}")

    # Step 3: Load and combine detectors for each half-day, then combine halves
    # Process CdTe
    cdte_halves = {}
    cdte_band_halves = {}
    for fg in selected:
        half = fg["half"]
        result = _combine_bands_for_half(fg, "cdte")
        if result is not None:
            cdte_halves[half] = result["total"]
            cdte_band_halves[half] = result["bands"]

    # Process CZT
    czt_halves = {}
    czt_band_halves = {}
    for fg in selected:
        half = fg["half"]
        result = _combine_bands_for_half(fg, "czt")
        if result is not None:
            czt_halves[half] = result["total"]
            czt_band_halves[half] = result["bands"]

    # Combine AM + PM
    cdte_full = _combine_halves(cdte_halves.get("AM"), cdte_halves.get("PM"))
    czt_full = _combine_halves(czt_halves.get("AM"), czt_halves.get("PM"))

    if cdte_full is None and czt_full is None:
        print(f"    [!] No usable data for {day_folder}")
        return None

    # Use CdTe as primary (broader low-energy coverage), CZT as secondary
    primary = cdte_full if cdte_full is not None else czt_full
    mjd = primary["mjd"]

    # Step 4: Convert MJD to seconds from day start
    mjd_day_start = np.floor(mjd[0])  # midnight MJD
    times = (mjd - mjd_day_start) * 86400.0

    # Step 5: Determine date string from MJD
    t_ref = Time(mjd_day_start, format="mjd")
    date_str = t_ref.datetime.strftime("%Y-%m-%d")

    # Step 6: Combine individual energy bands across halves
    band_data = {"cdte": {}, "czt": {}}
    for det_type, halves_dict in [("cdte", cdte_band_halves), ("czt", czt_band_halves)]:
        all_band_indices = set()
        for half_bands in halves_dict.values():
            all_band_indices.update(half_bands.keys())
        for bidx in sorted(all_band_indices):
            am_band = halves_dict.get("AM", {}).get(bidx)
            pm_band = halves_dict.get("PM", {}).get(bidx)
            combined = _combine_halves(am_band, pm_band)
            if combined is not None:
                band_data[det_type][bidx] = combined

    # Align CZT data to CdTe time grid if both exist
    czt_ctr = None
    czt_err = None
    if czt_full is not None and cdte_full is not None:
        # Interpolate CZT onto CdTe time grid
        czt_times = (czt_full["mjd"] - mjd_day_start) * 86400.0
        czt_ctr = np.interp(times, czt_times, czt_full["ctr"],
                            left=0.0, right=0.0)
        czt_err = np.interp(times, czt_times, czt_full["stat_err"],
                            left=0.0, right=0.0)
    elif czt_full is not None:
        czt_ctr = czt_full["ctr"]
        czt_err = czt_full["stat_err"]

    n_points = len(times)
    duration_h = (times[-1] - times[0]) / 3600.0
    cdte_mean = np.nanmean(primary["ctr"]) if cdte_full is not None else 0
    print(f"    {n_points:,} data points, {duration_h:.1f} hours")
    print(f"    CdTe total mean: {cdte_mean:.4f} cts/s")
    if czt_ctr is not None:
        print(f"    CZT total mean:  {np.nanmean(czt_ctr):.4f} cts/s")

    return {
        "times": times,
        "total_ctr": primary["ctr"],
        "total_err": primary["stat_err"],
        "czt_ctr": czt_ctr,
        "czt_err": czt_err,
        "band_data": band_data,
        "date_str": date_str,
        "mjd_ref": mjd_day_start,
        "mjd": mjd,
    }


# ═══════════════════════════════════════════════════════════════
# BACKGROUND ESTIMATION
# ═══════════════════════════════════════════════════════════════
def compute_background_fast(ctr, window=BG_WINDOW, percentile=BG_PERCENTILE):
    """
    Fast background estimation using a low percentile on sliding chunks.

    Adapted from SoLEXS version with lower percentile (20th vs 30th)
    because HEL1OS count rates are much lower and sparser.

    Parameters
    ----------
    ctr : array
        Count rate time series.
    window : int
        Window size in seconds for the background estimation.
    percentile : int
        Percentile to use (lower = more conservative background).

    Returns
    -------
    bg : array
        Background estimate, same length as ctr.
    """
    n = len(ctr)
    bg = np.full(n, np.nan)

    # Replace NaN/negative with 0 for estimation
    ctr_clean = np.nan_to_num(ctr, nan=0.0)
    ctr_clean = np.clip(ctr_clean, 0.0, None)

    # Process in chunks
    chunk_size = window // 2
    for i in range(0, n, chunk_size):
        start = max(0, i - window)
        end = min(n, i + window)
        chunk_pct = np.nanpercentile(ctr_clean[start:end], percentile)
        bg[i:min(i + chunk_size, n)] = chunk_pct

    # Interpolate any remaining NaNs
    mask = ~np.isnan(bg)
    if mask.sum() > 0:
        bg = np.interp(np.arange(n), np.where(mask)[0], bg[mask])

    # Apply smoothing to avoid step artifacts
    bg = uniform_filter1d(bg, size=window // 4)

    # Ensure background is never zero (avoid division by zero)
    bg = np.maximum(bg, 1e-6)

    return bg


# ═══════════════════════════════════════════════════════════════
# FLARE DETECTION
# ═══════════════════════════════════════════════════════════════
def detect_hel1os_flares(times, ctr, bg,
                         threshold=RISE_THRESHOLD,
                         min_duration=MIN_FLARE_DURATION,
                         min_peak=MIN_PEAK_COUNTS,
                         merge_gap=MERGE_GAP,
                         smoothing=SMOOTHING_WINDOW):
    """
    Detect flare events from an HEL1OS lightcurve using a threshold state machine.

    Same algorithm as SoLEXS detect_flares() but with parameters adapted
    for hard X-ray data:
      - Heavier smoothing (60s vs 30s)
      - Higher rise threshold (5.0× vs 3.0×)
      - Shorter minimum duration (20s vs 30s)
      - Lower minimum peak (1.0 cts/s vs 20 cts)

    Parameters
    ----------
    times : array
        Time in seconds from day start.
    ctr : array
        Count rate (cts/s).
    bg : array
        Background estimate.
    threshold : float
        Flare start threshold (counts > bg * threshold).
    min_duration : int
        Minimum flare duration in seconds.
    min_peak : float
        Minimum peak count rate (cts/s).
    merge_gap : int
        Merge detections within this many seconds.
    smoothing : int
        Smoothing window size in seconds.

    Returns
    -------
    list of dicts, each describing a detected flare event.
    """
    # Smooth the count rates
    ctr_clean = np.nan_to_num(ctr, nan=0.0)
    ctr_smooth = uniform_filter1d(ctr_clean, size=smoothing)

    # Detection threshold
    detection_level = bg * threshold

    # Where smoothed counts exceed threshold
    above = ctr_smooth > detection_level

    # Also require absolute minimum
    above = above & (ctr_smooth > min_peak / 2)

    # Find contiguous above-threshold regions
    flares = []
    
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            start_idx = i
            # Find where it drops below threshold
            end_idx = i
            while end_idx < n and above[end_idx]:
                end_idx += 1
                
            # Extend end into the decay tail
            while (end_idx < n - 1 and
                   ctr_smooth[end_idx] > bg[end_idx] * 1.5):
                end_idx += 1
                
            # Compute flare properties
            duration = end_idx - start_idx
            if end_idx > start_idx:
                peak_rel_idx = np.nanargmax(ctr_clean[start_idx:end_idx])
            else:
                i = end_idx + 1
                continue
                
            peak_idx = start_idx + peak_rel_idx
            peak_counts = ctr_clean[peak_idx]

            if duration >= min_duration and peak_counts >= min_peak:
                flares.append({
                    "start_idx": int(start_idx),
                    "peak_idx": int(peak_idx),
                    "end_idx": int(end_idx),
                    "start_time": float(times[start_idx]),
                    "peak_time": float(times[peak_idx]),
                    "end_time": float(times[min(end_idx, n - 1)]),
                    "peak_counts": float(peak_counts),
                    "mean_counts": float(np.nanmean(ctr_clean[start_idx:end_idx])),
                    "bg_at_peak": float(bg[peak_idx]),
                    "duration_sec": int(duration),
                    "rise_time_sec": int(peak_idx - start_idx),
                    "decay_time_sec": int(end_idx - peak_idx),
                    "intensity_ratio": float(peak_counts / max(bg[peak_idx], 1e-6)),
                })
            
            # Jump i forward to avoid rescanning
            i = end_idx
        else:
            i += 1

    # Merge flares that are close together (within merge_gap seconds)
    merged = []
    for flare in flares:
        if merged and (flare["start_time"] - merged[-1]["end_time"]) < merge_gap:
            # Merge into previous flare
            prev = merged[-1]
            prev["end_idx"] = flare["end_idx"]
            prev["end_time"] = flare["end_time"]
            prev["duration_sec"] = int(prev["end_idx"] - prev["start_idx"])
            if flare["peak_counts"] > prev["peak_counts"]:
                prev["peak_idx"] = flare["peak_idx"]
                prev["peak_time"] = flare["peak_time"]
                prev["peak_counts"] = flare["peak_counts"]
                prev["bg_at_peak"] = flare["bg_at_peak"]
                prev["intensity_ratio"] = flare["intensity_ratio"]
            prev["rise_time_sec"] = int(prev["peak_idx"] - prev["start_idx"])
            prev["decay_time_sec"] = int(prev["end_idx"] - prev["peak_idx"])
        else:
            merged.append(flare)

    # Recompute means at the very end to avoid O(N^2) complexity
    for prev in merged:
        s, e = prev["start_idx"], prev["end_idx"]
        prev["mean_counts"] = float(np.nanmean(ctr_clean[s:e]))

    return merged


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
def classify_flare_hel1os(peak_counts, bg_counts):
    """
    Classify flare intensity based on peak-to-background ratio.

    HEL1OS hard X-ray thresholds are higher than SoLEXS because
    hard X-ray emission during flares is much more impulsive and
    the background is much lower, leading to larger ratios for
    significant events.

    Approximate mapping (counts ratio -> GOES class):
      > 100× background -> X-class (very rare in hard X-ray)
      > 30×  background -> M-class
      > 10×  background -> C-class
      > 3×   background -> B-class
      otherwise          -> A-class
    """
    ratio = peak_counts / max(bg_counts, 1e-6)
    if ratio > 100:
        return "X"
    elif ratio > 30:
        return "M"
    elif ratio > 10:
        return "C"
    elif ratio > 3:
        return "B"
    else:
        return "A"


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def plot_day_with_flares(day_data, flares, date_str):
    """
    Plot a single day's HEL1OS lightcurve with detected flare markers.

    Produces a two-panel plot:
      - Top: CdTe total-band lightcurve with flare regions shaded
      - Bottom: CZT total-band lightcurve (if available)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    times = day_data["times"]
    ctr = day_data["total_ctr"]
    bg = compute_background_fast(np.nan_to_num(ctr, nan=0.0))
    czt_ctr = day_data.get("czt_ctr")

    has_czt = czt_ctr is not None and len(czt_ctr) > 0

    # Convert to hours for x-axis
    hours = times / 3600.0

    # Smooth for cleaner plot
    ctr_smooth = uniform_filter1d(np.nan_to_num(ctr, nan=0.0), size=60)

    n_panels = 2 if has_czt else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels),
                             sharex=True, squeeze=False)

    # ── Top panel: CdTe ──
    ax = axes[0, 0]
    # Use linear scale (counts are low), with floor at 0
    ax.plot(hours, ctr_smooth, linewidth=0.6, color="#E65100", alpha=0.9,
            label="CdTe total (1-min smooth)")
    ax.plot(hours, bg, linewidth=1, color="gray", alpha=0.6, linestyle="--",
            label="Background")
    ax.plot(hours, bg * RISE_THRESHOLD, linewidth=0.8, color="red", alpha=0.3,
            linestyle=":", label=f"{RISE_THRESHOLD}× threshold")

    # Mark flares
    flare_colors = {
        "A": "gray", "B": "#4CAF50", "C": "#FFC107", "M": "#FF9800", "X": "#F44336"
    }
    for flare in flares:
        start_h = flare["start_time"] / 3600.0
        peak_h = flare["peak_time"] / 3600.0
        end_h = flare["end_time"] / 3600.0
        fc = flare.get("flare_class", "?")
        color = flare_colors.get(fc, "purple")

        ax.axvspan(start_h, end_h, alpha=0.15, color=color)
        ax.plot(peak_h, flare["peak_counts"], "k*", markersize=12)

        peak_utc_short = flare.get("peak_utc", "").split(" ")[-1][:8] if "peak_utc" in flare else ""
        ax.annotate(
            f'{fc}-class\n{peak_utc_short}\n{flare["peak_counts"]:.2f} cts/s',
            xy=(peak_h, flare["peak_counts"]),
            xytext=(peak_h + 0.3, flare["peak_counts"] * 1.3 + 0.1),
            fontsize=7, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    ax.set_ylabel("CdTe Count Rate (cts/s)", fontsize=11)
    ax.set_title(
        f"HEL1OS Hard X-ray — {date_str} | {len(flares)} flare(s) detected",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Bottom panel: CZT (if available) ──
    if has_czt:
        ax2 = axes[1, 0]
        czt_smooth = uniform_filter1d(np.nan_to_num(czt_ctr, nan=0.0), size=60)
        ax2.plot(hours, czt_smooth, linewidth=0.6, color="#1565C0", alpha=0.9,
                 label="CZT total (1-min smooth)")

        # Mark flare regions from CdTe detection
        for flare in flares:
            start_h = flare["start_time"] / 3600.0
            end_h = flare["end_time"] / 3600.0
            fc = flare.get("flare_class", "?")
            color = flare_colors.get(fc, "purple")
            ax2.axvspan(start_h, end_h, alpha=0.15, color=color)

        ax2.set_ylabel("CZT Count Rate (cts/s)", fontsize=11)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

    # Shared x-axis
    axes[-1, 0].set_xlabel("Hours (UTC)", fontsize=11)
    axes[-1, 0].set_xlim(0, 24)
    axes[-1, 0].set_xticks(range(0, 25, 2))
    axes[-1, 0].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])

    plt.tight_layout()
    out_name = os.path.join(OUTPUT_DIR, f"hel1os_flares_{date_str}.png")
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved plot: {out_name}")


# ═══════════════════════════════════════════════════════════════
# PROCESS ALL DAYS
# ═══════════════════════════════════════════════════════════════
def process_all_dates():
    """
    Process all available HEL1OS dates and build the flare catalog.
    """
    all_flares = []
    daily_summaries = []

    if not os.path.isdir(HEL1OS_DIR):
        print(f"[!] HEL1OS data directory not found: {HEL1OS_DIR}")
        print(f"    Expected structure: {HEL1OS_DIR}/HLS_YYYYMMDD/")
        return all_flares, daily_summaries

    # Find all day folders
    day_folders = sorted([
        f for f in os.listdir(HEL1OS_DIR)
        if os.path.isdir(os.path.join(HEL1OS_DIR, f)) and f.startswith("HLS_")
    ])

    print(f"Found {len(day_folders)} days of HEL1OS data\n")
    print("=" * 80)

    for folder_name in day_folders:
        day_path = os.path.join(HEL1OS_DIR, folder_name)

        # Check that zip files have been extracted
        has_zips = any(f.endswith(".zip") for f in os.listdir(day_path))
        has_dirs = any(
            os.path.isdir(os.path.join(day_path, f)) and f.startswith("HLS_")
            for f in os.listdir(day_path)
        )
        if has_zips and not has_dirs:
            print(f"\n  [!] {folder_name}: Zip files found but NOT extracted — skipping")
            print(f"      Please extract zip files first.")
            continue

        # Load data
        day_data = load_hel1os_day(day_path)
        if day_data is None:
            continue

        date_str = day_data["date_str"]
        times = day_data["times"]
        ctr = day_data["total_ctr"]
        mjd_ref = day_data["mjd_ref"]

        # Clean NaNs
        ctr_clean = np.nan_to_num(ctr, nan=0.0)
        ctr_clean = np.clip(ctr_clean, 0.0, None)

        # Compute background
        print("    [DEBUG] Computing background...")
        bg = compute_background_fast(ctr_clean)

        # Detect flares
        print("    [DEBUG] Detecting flares...")
        flares = detect_hel1os_flares(times, ctr_clean, bg)

        # Classify each flare and add metadata
        for i, flare in enumerate(flares):
            flare_class = classify_flare_hel1os(flare["peak_counts"], flare["bg_at_peak"])
            flare["flare_class"] = flare_class
            flare["date"] = date_str
            flare["source"] = "HEL1OS"
            flare["detector"] = "CdTe"
            flare["flare_id"] = f"HEL_{date_str.replace('-', '')}_{i+1:03d}"

            # Convert seconds-from-day-start back to UTC
            peak_mjd = mjd_ref + flare["peak_time"] / 86400.0
            start_mjd = mjd_ref + flare["start_time"] / 86400.0
            end_mjd = mjd_ref + flare["end_time"] / 86400.0

            flare["peak_utc"] = Time(peak_mjd, format="mjd").iso
            flare["start_utc"] = Time(start_mjd, format="mjd").iso
            flare["end_utc"] = Time(end_mjd, format="mjd").iso

        # Summary
        n_flares = len(flares)
        classes = [f["flare_class"] for f in flares]
        class_summary = {c: classes.count(c) for c in sorted(set(classes))} if classes else {}
        print(f"    -> {n_flares} flare(s) detected: {class_summary}")

        daily_summaries.append({
            "date": date_str,
            "source": "HEL1OS",
            "detector": "CdTe",
            "n_flares": n_flares,
            "classes": class_summary,
            "max_ctr": float(np.nanmax(ctr_clean)),
            "mean_ctr": float(np.nanmean(ctr_clean)),
            "n_datapoints": len(times),
        })

        all_flares.extend(flares)

        # Plot this day
        try:
            plot_day_with_flares(day_data, flares, date_str)
        except Exception as e:
            print(f"    [!] Plot failed: {e}")

    print("\n" + "=" * 80)
    print(f"\nTotal HEL1OS flares detected: {len(all_flares)}")

    return all_flares, daily_summaries


# ═══════════════════════════════════════════════════════════════
# SAVE CATALOG
# ═══════════════════════════════════════════════════════════════
def save_catalog(all_flares, daily_summaries):
    """
    Save the HEL1OS flare catalog as JSON and CSV.
    Schema is compatible with SoLEXS master_flare_catalog.json,
    with additional 'source' and 'detector' fields.
    """
    # JSON catalog
    output = {
        "catalog": all_flares,
        "daily_summaries": daily_summaries,
        "config": {
            "smoothing_window": SMOOTHING_WINDOW,
            "bg_window": BG_WINDOW,
            "bg_percentile": BG_PERCENTILE,
            "rise_threshold": RISE_THRESHOLD,
            "min_flare_duration": MIN_FLARE_DURATION,
            "min_peak_counts": MIN_PEAK_COUNTS,
            "merge_gap": MERGE_GAP,
        },
        "instrument": "HEL1OS",
        "energy_range": "5-90 keV (CdTe total)",
    }
    json_path = "hel1os_flare_catalog.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[CATALOG] JSON catalog saved: {json_path}")

    # CSV catalog
    if all_flares:
        catalog_cols = [
            "flare_id", "date", "source", "detector", "flare_class",
            "start_utc", "peak_utc", "end_utc",
            "peak_counts", "mean_counts", "bg_at_peak",
            "duration_sec", "rise_time_sec", "decay_time_sec",
            "intensity_ratio",
        ]
        df = pd.DataFrame(all_flares)
        # Ensure all columns exist (add missing ones with NaN)
        for col in catalog_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[catalog_cols]
        csv_path = "hel1os_flare_catalog.csv"
        df.to_csv(csv_path, index=False)
        print(f"[CATALOG] CSV catalog saved: {csv_path} ({len(df)} flares)")

        # Print catalog summary
        print("\n" + "=" * 100)
        print("HEL1OS FLARE CATALOG — Hard X-ray Nowcasting")
        print("=" * 100)
        print(df.to_string(index=False))

        print("\n--- Summary by Class ---")
        for cls in ["X", "M", "C", "B", "A"]:
            n = (df["flare_class"] == cls).sum()
            if n > 0:
                print(f"  {cls}-class: {n} flare(s)")
    else:
        print("\n[WARNING] No flares detected in any HEL1OS data!")

    # Daily summary table
    if daily_summaries:
        print("\n--- Daily Summary ---")
        ds_df = pd.DataFrame(daily_summaries)
        print(ds_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[*] Solar Flare Detection - HEL1OS Hard X-ray Nowcasting Pipeline")
    print("    Instrument: HEL1OS (CdTe 5-90 keV, CZT 18-160 keV)")
    print("=" * 80)

    all_flares, daily_summaries = process_all_dates()
    save_catalog(all_flares, daily_summaries)

    print(f"\n[OK] Done! Check {OUTPUT_DIR}/hel1os_flares_*.png plots")
    print(f"     and hel1os_flare_catalog.csv / .json")
