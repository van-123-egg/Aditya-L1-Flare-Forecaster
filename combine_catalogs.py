"""
Combined Flare Catalog Builder — SoLEXS + HEL1OS
==================================================
Merges independent SoLEXS (soft X-ray) and HEL1OS (hard X-ray) flare catalogs
into a unified master combined catalog with cross-instrument matching.

Matching is performed using UTC peak times: flares from the two instruments
whose peaks are within MATCH_WINDOW seconds of each other are considered the
same physical event.  Unmatched flares are retained and flagged as single-
instrument detections.

Outputs:
  - master_combined_catalog.json
  - master_combined_catalog.csv
  - combined_catalog_overlap.png  (Venn-diagram-style bar chart)

Usage:
  python combine_catalogs.py [--match-window 120] [--solexs-catalog FILE]
                              [--hel1os-catalog FILE] [--output-dir DIR]
"""

import json
import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DEFAULT_MATCH_WINDOW = 120  # seconds — max |Deltat_peak| for a cross-match
SOLEXS_CATALOG_FILE = "master_flare_catalog.json"
HEL1OS_CATALOG_FILE = "hel1os_flare_catalog.json"

# Flare class ordering (A < B < C < M < X) for "higher class" logic
CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
CLASS_NAMES = {v: k for k, v in CLASS_ORDER.items()}


# ═══════════════════════════════════════════════════════════════
# UTC PARSING HELPERS
# ═══════════════════════════════════════════════════════════════
def parse_utc(utc_string):
    """
    Robustly parse a UTC string into a datetime object.

    Handles:
      - '2026-06-02 04:44:58.000'  (SoLEXS format, space separator)
      - '2026-06-02T04:44:58.000'  (ISO 8601 with T)
      - '2026-06-02T04:44:58'      (no fractional seconds)
    """
    if utc_string is None:
        return None
    s = str(utc_string).strip()
    # Replace space separator with 'T' for uniform ISO parsing
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Fallback: try common strptime patterns
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        warnings.warn(f"Could not parse UTC string: '{utc_string}'")
        return None


def utc_delta_seconds(dt1, dt2):
    """Return absolute time difference in seconds between two datetimes."""
    if dt1 is None or dt2 is None:
        return float("inf")
    return abs((dt1 - dt2).total_seconds())


# ═══════════════════════════════════════════════════════════════
# CATALOG LOADING
# ═══════════════════════════════════════════════════════════════
def load_catalog(filepath, instrument_name):
    """
    Load a flare catalog JSON file.

    Expected structure: {"catalog": [ {...}, {...}, ... ], ...}
    Returns the list of flare dicts, or an empty list on failure.
    """
    if not os.path.isfile(filepath):
        warnings.warn(
            f"[{instrument_name}] Catalog file not found: {filepath}  "
            f"— proceeding without {instrument_name} data."
        )
        return []

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        warnings.warn(
            f"[{instrument_name}] Failed to read {filepath}: {e}  "
            f"— proceeding without {instrument_name} data."
        )
        return []

    # Accept either {"catalog": [...]} or a bare list
    if isinstance(data, dict):
        catalog = data.get("catalog", [])
    elif isinstance(data, list):
        catalog = data
    else:
        warnings.warn(f"[{instrument_name}] Unexpected JSON structure in {filepath}")
        return []

    print(f"[OK] Loaded {instrument_name} catalog: {len(catalog)} flare(s) from {filepath}")
    return catalog


def enrich_with_parsed_times(catalog, instrument_name):
    """
    Pre-parse UTC strings into datetime objects for fast matching.
    Adds '_peak_dt', '_start_dt', '_end_dt' keys to each entry in-place.
    Returns count of entries that could NOT be parsed (for diagnostics).
    """
    n_bad = 0
    for entry in catalog:
        entry["_peak_dt"] = parse_utc(entry.get("peak_utc"))
        entry["_start_dt"] = parse_utc(entry.get("start_utc"))
        entry["_end_dt"] = parse_utc(entry.get("end_utc"))
        if entry["_peak_dt"] is None:
            n_bad += 1
    if n_bad:
        warnings.warn(
            f"[{instrument_name}] {n_bad} flare(s) have un-parsable peak_utc — "
            f"these will be treated as unmatched."
        )
    return n_bad


# ═══════════════════════════════════════════════════════════════
# MATCHING ALGORITHM
# ═══════════════════════════════════════════════════════════════
def match_catalogs(solexs_catalog, hel1os_catalog, match_window=DEFAULT_MATCH_WINDOW):
    """
    Cross-match SoLEXS and HEL1OS flares by UTC peak time proximity.

    Algorithm:
      1. Group HEL1OS flares by date for O(N) per-date lookup.
      2. For each SoLEXS flare, find the closest HEL1OS flare on the same
         date within ±match_window seconds.  If multiple HEL1OS flares are
         within the window, pick the closest in time.
      3. A given HEL1OS flare can only match ONE SoLEXS flare (the closest).
         If two SoLEXS flares both want the same HEL1OS flare, the closer
         pair wins and the other SoLEXS flare becomes SoLEXS-only.
      4. Any remaining unmatched HEL1OS flares become HEL1OS-only.

    Returns:
      matched_pairs : list of (solexs_entry, hel1os_entry, delta_sec)
      solexs_only   : list of solexs_entry
      hel1os_only   : list of hel1os_entry
    """
    # --- Group HEL1OS flares by date ---
    hel_by_date = {}
    for h in hel1os_catalog:
        dt = h.get("_peak_dt")
        if dt is None:
            continue
        date_key = dt.strftime("%Y-%m-%d")
        hel_by_date.setdefault(date_key, []).append(h)

    # --- For each SoLEXS flare, find best HEL1OS match ---
    # Store candidate matches as (slx_idx, hel_idx, delta_sec)
    # where indices reference the original catalog lists.
    slx_id_to_idx = {id(e): i for i, e in enumerate(solexs_catalog)}
    hel_id_to_idx = {id(e): i for i, e in enumerate(hel1os_catalog)}

    # Candidate matches: hel_idx -> list of (slx_idx, delta_sec)
    hel_candidates = {}

    for slx in solexs_catalog:
        slx_dt = slx.get("_peak_dt")
        if slx_dt is None:
            continue

        date_key = slx_dt.strftime("%Y-%m-%d")
        candidates = hel_by_date.get(date_key, [])

        # Also check neighboring dates (flare at 23:59 might match 00:01 next day)
        prev_date = (slx_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        next_date = (slx_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        for nd in [prev_date, next_date]:
            candidates = candidates + hel_by_date.get(nd, [])

        # Remove duplicates (if same-date candidates re-added via neighbor check)
        seen_ids = set()
        unique_candidates = []
        for c in candidates:
            if id(c) not in seen_ids:
                seen_ids.add(id(c))
                unique_candidates.append(c)
        candidates = unique_candidates

        best_hel = None
        best_delta = float("inf")
        for hel in candidates:
            delta = utc_delta_seconds(slx_dt, hel.get("_peak_dt"))
            if delta < match_window and delta < best_delta:
                best_hel = hel
                best_delta = delta

        if best_hel is not None:
            slx_idx = slx_id_to_idx[id(slx)]
            hel_idx = hel_id_to_idx[id(best_hel)]
            hel_candidates.setdefault(hel_idx, []).append((slx_idx, best_delta))

    # --- Resolve conflicts: each HEL1OS flare matches at most 1 SoLEXS flare ---
    matched_slx_indices = set()
    matched_hel_indices = set()
    matched_pairs = []

    for hel_idx, slx_candidates in hel_candidates.items():
        # Pick the SoLEXS flare closest in time
        slx_candidates.sort(key=lambda x: x[1])
        best_slx_idx, best_delta = slx_candidates[0]

        matched_pairs.append((
            solexs_catalog[best_slx_idx],
            hel1os_catalog[hel_idx],
            best_delta,
        ))
        matched_slx_indices.add(best_slx_idx)
        matched_hel_indices.add(hel_idx)

    # --- Collect unmatched ---
    solexs_only = [
        slx for i, slx in enumerate(solexs_catalog)
        if i not in matched_slx_indices
    ]
    hel1os_only = [
        hel for i, hel in enumerate(hel1os_catalog)
        if i not in matched_hel_indices
    ]

    return matched_pairs, solexs_only, hel1os_only


# ═══════════════════════════════════════════════════════════════
# COMBINED ENTRY BUILDER
# ═══════════════════════════════════════════════════════════════
def higher_class(cls1, cls2):
    """
    Return the higher flare class (e.g. 'M' > 'C').
    If one is None/empty, return the other.
    """
    if not cls1:
        return cls2 or "?"
    if not cls2:
        return cls1 or "?"
    o1 = CLASS_ORDER.get(cls1, -1)
    o2 = CLASS_ORDER.get(cls2, -1)
    return cls1 if o1 >= o2 else cls2


def _safe_float(val, default=None):
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_combined_entry(slx, hel, detection_source, date_str, seq_num):
    """
    Build a single combined catalog entry.

    Parameters
    ----------
    slx : dict or None — SoLEXS flare entry
    hel : dict or None — HEL1OS flare entry
    detection_source : str — 'SoLEXS+HEL1OS', 'SoLEXS_only', or 'HEL1OS_only'
    date_str : str — 'YYYYMMDD'
    seq_num : int — sequence number for this date
    """
    combined_id = f"CMB_{date_str}_{seq_num:03d}"

    # Determine confidence
    if detection_source == "SoLEXS+HEL1OS":
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    # SoLEXS prefixed fields
    slx_peak_utc = slx.get("peak_utc") if slx else None
    slx_peak_counts = _safe_float(slx.get("peak_counts")) if slx else None
    slx_bg = _safe_float(slx.get("bg_at_peak")) if slx else None
    slx_duration = _safe_float(slx.get("duration_sec")) if slx else None
    slx_class = slx.get("flare_class") if slx else None
    slx_flare_id = slx.get("flare_id") if slx else None
    slx_intensity_ratio = _safe_float(slx.get("intensity_ratio")) if slx else None
    slx_start_utc = slx.get("start_utc") if slx else None
    slx_end_utc = slx.get("end_utc") if slx else None
    slx_rise = _safe_float(slx.get("rise_time_sec")) if slx else None
    slx_decay = _safe_float(slx.get("decay_time_sec")) if slx else None

    # HEL1OS prefixed fields
    # The HEL1OS catalog uses 'peak_ctr' (count rate) instead of 'peak_counts'.
    # Also accept 'peak_counts' as fallback for flexibility.
    hel_peak_utc = hel.get("peak_utc") if hel else None
    hel_peak_ctr = _safe_float(
        hel.get("peak_ctr", hel.get("peak_counts"))
    ) if hel else None
    hel_bg = _safe_float(hel.get("bg_at_peak")) if hel else None
    hel_duration = _safe_float(hel.get("duration_sec")) if hel else None
    hel_class = hel.get("flare_class") if hel else None
    hel_flare_id = hel.get("flare_id") if hel else None
    hel_intensity_ratio = _safe_float(hel.get("intensity_ratio")) if hel else None
    hel_start_utc = hel.get("start_utc") if hel else None
    hel_end_utc = hel.get("end_utc") if hel else None
    hel_rise = _safe_float(hel.get("rise_time_sec")) if hel else None
    hel_decay = _safe_float(hel.get("decay_time_sec")) if hel else None
    hel_detector = hel.get("detector") if hel else None
    hel_energy_band = hel.get("energy_band") if hel else None

    # Combined class: take the higher
    combined_class = higher_class(slx_class, hel_class)

    # Soft-hard delay: positive means hard peaks first (expected physically)
    soft_hard_delay_sec = None
    if slx and hel:
        slx_dt = slx.get("_peak_dt") or parse_utc(slx_peak_utc)
        hel_dt = hel.get("_peak_dt") or parse_utc(hel_peak_utc)
        if slx_dt and hel_dt:
            # Positive = HEL1OS peak happens BEFORE SoLEXS peak
            soft_hard_delay_sec = round((slx_dt - hel_dt).total_seconds(), 3)

    # Determine the display date (YYYY-MM-DD)
    if slx:
        display_date = slx.get("date", date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8])
    elif hel:
        display_date = hel.get("date", date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8])
    else:
        display_date = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8]

    entry = OrderedDict([
        ("combined_id",           combined_id),
        ("date",                  display_date),
        ("detection_source",      detection_source),
        ("confidence",            confidence),
        ("combined_class",        combined_class),
        ("soft_hard_delay_sec",   soft_hard_delay_sec),
        # -- SoLEXS fields --
        ("slx_flare_id",          slx_flare_id),
        ("slx_peak_utc",          slx_peak_utc),
        ("slx_start_utc",         slx_start_utc),
        ("slx_end_utc",           slx_end_utc),
        ("slx_peak_counts",       slx_peak_counts),
        ("slx_bg",                slx_bg),
        ("slx_duration_sec",      slx_duration),
        ("slx_class",             slx_class),
        ("slx_intensity_ratio",   slx_intensity_ratio),
        ("slx_rise_time_sec",     slx_rise),
        ("slx_decay_time_sec",    slx_decay),
        # -- HEL1OS fields --
        ("hel_flare_id",          hel_flare_id),
        ("hel_peak_utc",          hel_peak_utc),
        ("hel_start_utc",         hel_start_utc),
        ("hel_end_utc",           hel_end_utc),
        ("hel_peak_ctr",          hel_peak_ctr),
        ("hel_bg",                hel_bg),
        ("hel_duration_sec",      hel_duration),
        ("hel_class",             hel_class),
        ("hel_intensity_ratio",   hel_intensity_ratio),
        ("hel_rise_time_sec",     hel_rise),
        ("hel_decay_time_sec",    hel_decay),
        ("hel_detector",          hel_detector),
        ("hel_energy_band",       hel_energy_band),
    ])
    return entry


# ═══════════════════════════════════════════════════════════════
# COMBINED CATALOG ASSEMBLY
# ═══════════════════════════════════════════════════════════════
def build_combined_catalog(matched_pairs, solexs_only, hel1os_only):
    """
    Assemble the full combined catalog from matched and unmatched flares.
    Combined IDs are numbered sequentially per date, sorted by peak UTC.
    """
    # Collect all entries with a sortable UTC key
    raw_entries = []  # (peak_datetime, detection_source, slx_dict, hel_dict)

    for slx, hel, delta in matched_pairs:
        dt = slx.get("_peak_dt") or hel.get("_peak_dt")
        raw_entries.append((dt, "SoLEXS+HEL1OS", slx, hel))

    for slx in solexs_only:
        dt = slx.get("_peak_dt")
        raw_entries.append((dt, "SoLEXS_only", slx, None))

    for hel in hel1os_only:
        dt = hel.get("_peak_dt")
        raw_entries.append((dt, "HEL1OS_only", None, hel))

    # Sort by peak time (put None-time entries at end)
    raw_entries.sort(key=lambda x: x[0] or datetime.max)

    # Assign combined IDs per date
    date_counters = Counter()
    combined_catalog = []

    for peak_dt, detection_source, slx, hel in raw_entries:
        if peak_dt is not None:
            date_str = peak_dt.strftime("%Y%m%d")
        elif slx and slx.get("date"):
            date_str = slx["date"].replace("-", "")
        elif hel and hel.get("date"):
            date_str = hel["date"].replace("-", "")
        else:
            date_str = "00000000"

        date_counters[date_str] += 1
        seq_num = date_counters[date_str]

        entry = build_combined_entry(slx, hel, detection_source, date_str, seq_num)
        combined_catalog.append(entry)

    return combined_catalog


# ═══════════════════════════════════════════════════════════════
# SUMMARY & REPORTING
# ═══════════════════════════════════════════════════════════════
def print_summary(combined_catalog, n_slx_input, n_hel_input):
    """Print a human-readable summary table to stdout."""
    n_total = len(combined_catalog)
    n_both = sum(1 for e in combined_catalog if e["detection_source"] == "SoLEXS+HEL1OS")
    n_slx_only = sum(1 for e in combined_catalog if e["detection_source"] == "SoLEXS_only")
    n_hel_only = sum(1 for e in combined_catalog if e["detection_source"] == "HEL1OS_only")

    print("\n" + "=" * 70)
    print("  COMBINED FLARE CATALOG — SUMMARY")
    print("=" * 70)
    print(f"  SoLEXS input flares  : {n_slx_input:5d}")
    print(f"  HEL1OS input flares  : {n_hel_input:5d}")
    print(f"  ---------------------------------")
    print(f"  Cross-matched (both) : {n_both:5d}  (confidence: HIGH)")
    print(f"  SoLEXS only          : {n_slx_only:5d}  (confidence: MEDIUM)")
    print(f"  HEL1OS only          : {n_hel_only:5d}  (confidence: MEDIUM)")
    print(f"  ---------------------------------")
    print(f"  Total combined       : {n_total:5d}")

    # Delay statistics for matched flares
    delays = [
        e["soft_hard_delay_sec"] for e in combined_catalog
        if e["soft_hard_delay_sec"] is not None
    ]
    if delays:
        delays_arr = np.array(delays)
        print(f"\n  Soft->Hard delay (matched flares):")
        print(f"    Mean  : {np.mean(delays_arr):+.1f} sec")
        print(f"    Median: {np.median(delays_arr):+.1f} sec")
        print(f"    Range : [{np.min(delays_arr):+.1f}, {np.max(delays_arr):+.1f}] sec")
        n_hard_first = np.sum(delays_arr > 0)
        print(f"    Hard peaks first (Delta>0): {n_hard_first}/{len(delays_arr)} "
              f"({100*n_hard_first/len(delays_arr):.0f}%)")

    # Per-class breakdown
    print(f"\n  {'Class':<8s} {'Both':>6s} {'SLX-only':>9s} {'HEL-only':>9s} {'Total':>6s}")
    print(f"  {'-'*40}")
    for cls in ["X", "M", "C", "B", "A", "?"]:
        n_b = sum(1 for e in combined_catalog
                  if e["combined_class"] == cls and e["detection_source"] == "SoLEXS+HEL1OS")
        n_s = sum(1 for e in combined_catalog
                  if e["combined_class"] == cls and e["detection_source"] == "SoLEXS_only")
        n_h = sum(1 for e in combined_catalog
                  if e["combined_class"] == cls and e["detection_source"] == "HEL1OS_only")
        n_t = n_b + n_s + n_h
        if n_t > 0:
            print(f"  {cls:<8s} {n_b:>6d} {n_s:>9d} {n_h:>9d} {n_t:>6d}")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def plot_overlap_chart(combined_catalog, output_dir="."):
    """
    Create a Venn-diagram-style grouped bar chart showing detection overlap.
    Shows per-class counts colour-coded by detection source.
    """
    classes = ["A", "B", "C", "M", "X"]
    sources = ["SoLEXS+HEL1OS", "SoLEXS_only", "HEL1OS_only"]
    colors = {"SoLEXS+HEL1OS": "#4CAF50", "SoLEXS_only": "#2196F3", "HEL1OS_only": "#FF9800"}
    labels = {"SoLEXS+HEL1OS": "Both instruments", "SoLEXS_only": "SoLEXS only", "HEL1OS_only": "HEL1OS only"}

    # Count per class × source
    counts_matrix = {}
    for src in sources:
        counts_matrix[src] = []
        for cls in classes:
            n = sum(1 for e in combined_catalog
                    if e["combined_class"] == cls and e["detection_source"] == src)
            counts_matrix[src].append(n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 2]})

    # -- Panel 1: Grouped bar chart per class --
    ax = axes[0]
    x = np.arange(len(classes))
    bar_width = 0.25

    for i, src in enumerate(sources):
        ax.bar(x + i * bar_width, counts_matrix[src], bar_width,
               color=colors[src], label=labels[src], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Flare Class", fontsize=12)
    ax.set_ylabel("Number of Flares", fontsize=12)
    ax.set_title("Detection Overlap by Flare Class", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(classes, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for i, src in enumerate(sources):
        for j, v in enumerate(counts_matrix[src]):
            if v > 0:
                ax.text(x[j] + i * bar_width, v + 0.3, str(v),
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=colors[src])

    # -- Panel 2: Venn-style summary circles --
    ax2 = axes[1]
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Detection Summary", fontsize=14, fontweight="bold")

    n_both = sum(1 for e in combined_catalog if e["detection_source"] == "SoLEXS+HEL1OS")
    n_slx = sum(1 for e in combined_catalog if e["detection_source"] == "SoLEXS_only")
    n_hel = sum(1 for e in combined_catalog if e["detection_source"] == "HEL1OS_only")

    # Draw overlapping circles
    circle1 = plt.Circle((-0.7, 0), 1.5, alpha=0.25, color="#2196F3", linewidth=2, edgecolor="#1565C0")
    circle2 = plt.Circle((0.7, 0), 1.5, alpha=0.25, color="#FF9800", linewidth=2, edgecolor="#E65100")
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)

    # Labels
    ax2.text(-1.6, 0, f"{n_slx}\nSoLEXS\nonly", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#1565C0")
    ax2.text(0, 0, f"{n_both}\nBoth", ha="center", va="center",
             fontsize=16, fontweight="bold", color="#2E7D32")
    ax2.text(1.6, 0, f"{n_hel}\nHEL1OS\nonly", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#E65100")

    # Instrument labels
    ax2.text(-1.5, 2.0, "SoLEXS (soft X-ray)", ha="center", fontsize=11, color="#1565C0")
    ax2.text(1.5, 2.0, "HEL1OS (hard X-ray)", ha="center", fontsize=11, color="#E65100")
    ax2.text(0, -2.2, f"Total: {n_both + n_slx + n_hel} unique flare events",
             ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    outpath = os.path.join(output_dir, "combined_catalog_overlap.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved overlap chart: {outpath}")


# ═══════════════════════════════════════════════════════════════
# SAVE CATALOG
# ═══════════════════════════════════════════════════════════════
def save_catalog(combined_catalog, output_dir="."):
    """Save the combined catalog as JSON and CSV."""
    json_path = os.path.join(output_dir, "master_combined_catalog.json")
    csv_path = os.path.join(output_dir, "master_combined_catalog.csv")

    # -- JSON --
    # Convert OrderedDicts to regular dicts for cleaner JSON
    catalog_for_json = [dict(e) for e in combined_catalog]
    output = {
        "catalog": catalog_for_json,
        "metadata": {
            "description": "Combined SoLEXS + HEL1OS flare catalog",
            "match_window_sec": DEFAULT_MATCH_WINDOW,
            "generated_utc": datetime.utcnow().isoformat(),
            "n_total": len(combined_catalog),
            "n_both": sum(1 for e in combined_catalog if e["detection_source"] == "SoLEXS+HEL1OS"),
            "n_solexs_only": sum(1 for e in combined_catalog if e["detection_source"] == "SoLEXS_only"),
            "n_hel1os_only": sum(1 for e in combined_catalog if e["detection_source"] == "HEL1OS_only"),
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[SAVE] JSON catalog: {json_path}")

    # -- CSV --
    df = pd.DataFrame(combined_catalog)
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] CSV  catalog: {csv_path} ({len(df)} rows)")

    return json_path, csv_path


# ═══════════════════════════════════════════════════════════════
# CLEANUP HELPER
# ═══════════════════════════════════════════════════════════════
def strip_internal_keys(catalog):
    """Remove internal '_*' keys (parsed datetimes) before serialisation."""
    for entry in catalog:
        for key in list(entry.keys()):
            if key.startswith("_"):
                del entry[key]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Merge SoLEXS and HEL1OS flare catalogs into a combined catalog."
    )
    parser.add_argument(
        "--match-window", type=float, default=DEFAULT_MATCH_WINDOW,
        help=f"Maximum |Deltapeak_utc| in seconds for a cross-match (default: {DEFAULT_MATCH_WINDOW})"
    )
    parser.add_argument(
        "--solexs-catalog", type=str, default=SOLEXS_CATALOG_FILE,
        help=f"Path to SoLEXS catalog JSON (default: {SOLEXS_CATALOG_FILE})"
    )
    parser.add_argument(
        "--hel1os-catalog", type=str, default=HEL1OS_CATALOG_FILE,
        help=f"Path to HEL1OS catalog JSON (default: {HEL1OS_CATALOG_FILE})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output files (default: current directory)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  COMBINE CATALOGS — SoLEXS + HEL1OS Flare Catalog Merger")
    print("=" * 70)
    print(f"  Match window : {args.match_window:.0f} seconds")
    print(f"  SoLEXS input : {args.solexs_catalog}")
    print(f"  HEL1OS input : {args.hel1os_catalog}")
    print(f"  Output dir   : {args.output_dir}")
    print()

    # -- 1. Load catalogs --
    solexs_catalog = load_catalog(args.solexs_catalog, "SoLEXS")
    hel1os_catalog = load_catalog(args.hel1os_catalog, "HEL1OS")

    if not solexs_catalog and not hel1os_catalog:
        print("\n[ERROR] No catalog data available from either instrument. Exiting.")
        sys.exit(1)

    # -- 2. Parse UTC timestamps --
    enrich_with_parsed_times(solexs_catalog, "SoLEXS")
    enrich_with_parsed_times(hel1os_catalog, "HEL1OS")

    n_slx_input = len(solexs_catalog)
    n_hel_input = len(hel1os_catalog)

    # -- 3. Cross-match --
    if solexs_catalog and hel1os_catalog:
        print(f"\n[MATCH] Cross-matching {n_slx_input} SoLEXS × {n_hel_input} HEL1OS flares "
              f"(window={args.match_window:.0f}s)...")
        matched_pairs, solexs_only, hel1os_only = match_catalogs(
            solexs_catalog, hel1os_catalog, match_window=args.match_window
        )
        print(f"  -> Matched pairs : {len(matched_pairs)}")
        print(f"  -> SoLEXS only   : {len(solexs_only)}")
        print(f"  -> HEL1OS only   : {len(hel1os_only)}")
    elif solexs_catalog:
        print("\n[INFO] No HEL1OS catalog — all SoLEXS flares marked as SoLEXS_only.")
        matched_pairs = []
        solexs_only = solexs_catalog
        hel1os_only = []
    else:
        print("\n[INFO] No SoLEXS catalog — all HEL1OS flares marked as HEL1OS_only.")
        matched_pairs = []
        solexs_only = []
        hel1os_only = hel1os_catalog

    # -- 4. Build combined catalog --
    combined_catalog = build_combined_catalog(matched_pairs, solexs_only, hel1os_only)

    # -- 5. Summary --
    print_summary(combined_catalog, n_slx_input, n_hel_input)

    # -- 6. Remove internal keys before saving --
    # (The _peak_dt etc. are datetime objects, not JSON-serialisable)
    # Note: we already copied what we need into the combined entries.
    strip_internal_keys(solexs_catalog)
    strip_internal_keys(hel1os_catalog)

    # -- 7. Save --
    os.makedirs(args.output_dir, exist_ok=True)
    save_catalog(combined_catalog, output_dir=args.output_dir)

    # -- 8. Plot --
    try:
        plot_overlap_chart(combined_catalog, output_dir=args.output_dir)
    except Exception as e:
        warnings.warn(f"[PLOT] Could not generate overlap chart: {e}")

    print("\n[OK] Combined catalog generation complete.")


if __name__ == "__main__":
    main()
