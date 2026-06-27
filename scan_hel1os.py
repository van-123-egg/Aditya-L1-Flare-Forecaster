"""
scan_hel1os.py — Scan extracted HEL1OS data and produce a summary table.

For each day folder in hel1os/:
  1. Finds all extracted folders and lists available lightcurve FITS files.
  2. For each lightcurve file, reads the total-band HDU (HDU[5]) and prints:
     number of rows, time span, min/max/mean CTR.
  3. Checks if AM and PM halves are both present.
  4. Prints a final table showing data availability per day.

Usage:
    python scan_hel1os.py
"""

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
from collections import defaultdict

# ── Configuration ──
HEL1OS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hel1os")

# Lightcurve file names and their total-band HDU index
# CdTe total band: HDU[5] = "1.8-90 keV"
# CZT total band:  HDU[5] = "18-160 keV"
LIGHTCURVE_FILES = {
    "lightcurve_cdte1.fits": {"detector": "CdTe", "unit": 1, "total_hdu": 5, "band": "1.8-90 keV"},
    "lightcurve_cdte2.fits": {"detector": "CdTe", "unit": 2, "total_hdu": 5, "band": "1.8-90 keV"},
    "lightcurve_czt1.fits":  {"detector": "CZT",  "unit": 1, "total_hdu": 5, "band": "18-160 keV"},
    "lightcurve_czt2.fits":  {"detector": "CZT",  "unit": 2, "total_hdu": 5, "band": "18-160 keV"},
}


def classify_half_day(folder_name):
    """
    Classify a zip/folder as AM or PM based on the start-time field in the name.
    Naming: HLS_YYYYMMDD_HHMMSS_*sec_lev1_V*
    AM: starts before 12:00:00 (HH < 12)
    PM: starts at or after 12:00:00 (HH >= 12)
    
    Special case: some folders span midnight — we classify by the start time.
    """
    parts = folder_name.split("_")
    if len(parts) < 3:
        return "??"
    time_str = parts[2]  # HHMMSS
    try:
        hh = int(time_str[:2])
        return "AM" if hh < 12 else "PM"
    except (ValueError, IndexError):
        return "??"


def find_lightcurves_in_extracted(extracted_path):
    """
    Walk the extracted folder to find lightcurve FITS files.
    Returns dict: {filename: full_path}
    """
    found = {}
    for root, dirs, files in os.walk(extracted_path):
        for f in files:
            if f in LIGHTCURVE_FILES:
                found[f] = os.path.join(root, f)
    return found


def read_total_band_stats(fits_path, lc_info):
    """
    Read the total-band HDU from a lightcurve FITS file.
    Returns dict with stats or None on failure.
    """
    try:
        with fits.open(fits_path) as hdul:
            hdu_idx = lc_info["total_hdu"]
            
            if hdu_idx >= len(hdul):
                return {"error": f"HDU[{hdu_idx}] not found (only {len(hdul)} HDUs)"}
            
            data = hdul[hdu_idx].data
            header = hdul[hdu_idx].header
            
            if data is None or len(data) == 0:
                return {"error": "No data in total-band HDU"}
            
            n_rows = len(data)
            
            # Read MJD for time span
            mjd = data["MJD"]
            mjd_min = float(np.nanmin(mjd))
            mjd_max = float(np.nanmax(mjd))
            
            # Convert MJD to ISO times for display
            t_start = Time(mjd_min, format='mjd')
            t_end = Time(mjd_max, format='mjd')
            duration_hours = (mjd_max - mjd_min) * 24.0
            
            # Read CTR (count rate)
            ctr = data["CTR"].astype(float)
            # Filter out NaNs for stats
            valid_mask = np.isfinite(ctr)
            n_valid = int(np.sum(valid_mask))
            
            if n_valid == 0:
                return {
                    "n_rows": n_rows, "n_valid": 0,
                    "t_start": t_start.iso, "t_end": t_end.iso,
                    "duration_hours": duration_hours,
                    "min_ctr": float("nan"), "max_ctr": float("nan"), 
                    "mean_ctr": float("nan"),
                    "band_name": header.get("EXTNAME", lc_info["band"]),
                }
            
            ctr_valid = ctr[valid_mask]
            
            return {
                "n_rows": n_rows,
                "n_valid": n_valid,
                "t_start": t_start.iso,
                "t_end": t_end.iso,
                "duration_hours": duration_hours,
                "min_ctr": float(np.min(ctr_valid)),
                "max_ctr": float(np.max(ctr_valid)),
                "mean_ctr": float(np.mean(ctr_valid)),
                "band_name": header.get("EXTNAME", lc_info["band"]),
            }
    except Exception as e:
        return {"error": str(e)}


def scan_all():
    """Main scan routine."""
    if not os.path.isdir(HEL1OS_DIR):
        print(f"[ERROR] HEL1OS directory not found: {HEL1OS_DIR}")
        return
    
    # Find all day folders
    day_folders = sorted([
        d for d in os.listdir(HEL1OS_DIR)
        if os.path.isdir(os.path.join(HEL1OS_DIR, d)) and d.startswith("HLS_")
    ])
    
    print(f"Scanning {len(day_folders)} day folders in {HEL1OS_DIR}")
    print("=" * 100)
    
    # Collect per-day summary for final table
    day_summary = []
    
    for day_folder in day_folders:
        day_path = os.path.join(HEL1OS_DIR, day_folder)
        date_str = day_folder[4:]  # YYYYMMDD
        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        print(f"\n{'='*100}")
        print(f"DAY: {date_fmt} ({day_folder})")
        print(f"{'='*100}")
        
        # Find all extracted folders (directories that aren't .zip)
        extracted_folders = sorted([
            d for d in os.listdir(day_path)
            if os.path.isdir(os.path.join(day_path, d))
        ])
        
        if not extracted_folders:
            print("  [WARN] No extracted folders found — zips may not be extracted yet.")
            day_summary.append({
                "date": date_fmt, "n_folders": 0,
                "has_am": False, "has_pm": False,
                "cdte_lc": 0, "czt_lc": 0,
                "total_rows": 0, "mean_ctr_cdte": None, "mean_ctr_czt": None,
            })
            continue
        
        am_present = False
        pm_present = False
        day_cdte_lc = 0
        day_czt_lc = 0
        day_total_rows = 0
        day_cdte_ctr_values = []
        day_czt_ctr_values = []
        
        # Group folders by version — prefer V111 (or pick highest version)
        # But scan ALL folders, noting their half-day and version
        for ext_folder in extracted_folders:
            ext_path = os.path.join(day_path, ext_folder)
            half = classify_half_day(ext_folder)
            
            if half == "AM":
                am_present = True
            elif half == "PM":
                pm_present = True
            
            # Determine version string
            version = "?"
            for part in ext_folder.split("_"):
                if part.startswith("V") and part[1:].isdigit():
                    version = part
            
            print(f"\n  Folder: {ext_folder}")
            print(f"  Half: {half}  Version: {version}")
            
            # Find lightcurve files
            lc_files = find_lightcurves_in_extracted(ext_path)
            
            if not lc_files:
                print("    No lightcurve FITS files found")
                continue
            
            print(f"    Found {len(lc_files)} lightcurve file(s):")
            
            for lc_name in sorted(lc_files.keys()):
                lc_path = lc_files[lc_name]
                lc_info = LIGHTCURVE_FILES[lc_name]
                
                stats = read_total_band_stats(lc_path, lc_info)
                
                if "error" in stats:
                    print(f"      {lc_name}: [ERROR] {stats['error']}")
                    continue
                
                # Track counts
                if lc_info["detector"] == "CdTe":
                    day_cdte_lc += 1
                    day_cdte_ctr_values.append(stats["mean_ctr"])
                else:
                    day_czt_lc += 1
                    day_czt_ctr_values.append(stats["mean_ctr"])
                
                day_total_rows += stats["n_rows"]
                
                print(f"      {lc_name} [{lc_info['detector']}{lc_info['unit']}] "
                      f"total band={stats['band_name']}")
                print(f"        Rows: {stats['n_rows']:,}  "
                      f"Valid: {stats['n_valid']:,}  "
                      f"Duration: {stats['duration_hours']:.2f} h")
                print(f"        Time: {stats['t_start']} -> {stats['t_end']}")
                print(f"        CTR:  min={stats['min_ctr']:.6f}  "
                      f"max={stats['max_ctr']:.6f}  "
                      f"mean={stats['mean_ctr']:.6f} cts/s")
        
        day_summary.append({
            "date": date_fmt,
            "n_folders": len(extracted_folders),
            "has_am": am_present,
            "has_pm": pm_present,
            "cdte_lc": day_cdte_lc,
            "czt_lc": day_czt_lc,
            "total_rows": day_total_rows,
            "mean_ctr_cdte": np.mean(day_cdte_ctr_values) if day_cdte_ctr_values else None,
            "mean_ctr_czt": np.mean(day_czt_ctr_values) if day_czt_ctr_values else None,
        })
    
    # ── Final Summary Table ──
    print("\n\n" + "=" * 120)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 120)
    
    header = (f"{'Date':<12} {'Folders':>7} {'AM':>4} {'PM':>4} "
              f"{'CdTe LC':>8} {'CZT LC':>8} "
              f"{'Total Rows':>12} "
              f"{'CdTe mean CTR':>15} {'CZT mean CTR':>15} "
              f"{'Complete':>10}")
    print(header)
    print("-" * 120)
    
    complete_days = 0
    total_days = len(day_summary)
    
    for ds in day_summary:
        am_str = "Y" if ds["has_am"] else "N"
        pm_str = "Y" if ds["has_pm"] else "N"
        complete = ds["has_am"] and ds["has_pm"] and ds["cdte_lc"] > 0 and ds["czt_lc"] > 0
        if complete:
            complete_days += 1
        complete_str = "YES" if complete else "NO"
        
        cdte_ctr_str = f"{ds['mean_ctr_cdte']:.6f}" if ds['mean_ctr_cdte'] is not None else "N/A"
        czt_ctr_str = f"{ds['mean_ctr_czt']:.6f}" if ds['mean_ctr_czt'] is not None else "N/A"
        
        print(f"{ds['date']:<12} {ds['n_folders']:>7} {am_str:>4} {pm_str:>4} "
              f"{ds['cdte_lc']:>8} {ds['czt_lc']:>8} "
              f"{ds['total_rows']:>12,} "
              f"{cdte_ctr_str:>15} {czt_ctr_str:>15} "
              f"{complete_str:>10}")
    
    print("-" * 120)
    print(f"\nTotal days: {total_days}")
    print(f"Complete days (AM+PM + CdTe + CZT): {complete_days}/{total_days}")
    print("\nDONE — HEL1OS scan complete.")


if __name__ == "__main__":
    scan_all()
