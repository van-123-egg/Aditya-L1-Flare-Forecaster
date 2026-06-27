"""
Explore HEL1OS Level-1 FITS data structure.
Inspects lightcurve, spectrum, GTI, and event files for both CdTe and CZT detectors.
"""
from astropy.io import fits
import numpy as np
import os
import glob


# ── Find an extracted HEL1OS folder ──
HEL1OS_DIR = "hel1os"
extracted = None
for day_folder in sorted(os.listdir(HEL1OS_DIR)):
    day_path = os.path.join(HEL1OS_DIR, day_folder)
    if not os.path.isdir(day_path):
        continue
    for item in os.listdir(day_path):
        item_path = os.path.join(day_path, item)
        if os.path.isdir(item_path) and not item.endswith(".zip"):
            # Check if it has FITS files inside (may be nested 2026/06/DD/...)
            fits_files = []
            for root, dirs, files in os.walk(item_path):
                fits_files.extend([os.path.join(root, f) for f in files if f.endswith(".fits")])
            if fits_files:
                extracted = item_path
                break
    if extracted:
        break

if not extracted:
    print("[ERROR] No extracted HEL1OS folder found! Please extract at least one zip.")
    exit(1)

# Find all FITS files recursively
fits_files = []
for root, dirs, files in os.walk(extracted):
    for f in files:
        fits_files.extend([(f, os.path.join(root, f))])

print(f"Extracted folder: {extracted}")
print(f"Found {len(fits_files)} files:")
for name, path in sorted(fits_files):
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {name:45s} {size_mb:8.1f} MB")

print("\n" + "=" * 80)

# ── Inspect each type of FITS file ──
for name, path in sorted(fits_files):
    if not name.endswith(".fits"):
        print(f"\n--- {name} (non-FITS, skipping) ---")
        continue

    print(f"\n{'=' * 80}")
    print(f"FILE: {name}")
    print(f"{'=' * 80}")

    try:
        hdul = fits.open(path)
        print(f"HDU count: {len(hdul)}")
        hdul.info()

        for i, hdu in enumerate(hdul):
            print(f"\n  --- HDU[{i}]: {hdu.name} ({type(hdu).__name__}) ---")

            # Print key header values
            h = hdu.header
            important_keys = [
                "EXTNAME", "NAXIS", "NAXIS1", "NAXIS2",
                "TSTART", "TSTOP", "TIMEDEL", "TIMEZERO",
                "MJDREFI", "MJDREFF", "TELESCOP", "INSTRUME",
                "DETNAM", "FILTER", "CONTENT", "HDUCLAS1",
                "TFIELDS",
            ]
            for key in important_keys:
                if key in h:
                    print(f"    {key:12s} = {h[key]}")

            # Print columns for binary tables
            if hasattr(hdu, 'columns') and hdu.columns is not None:
                print(f"\n    Columns ({len(hdu.columns)}):")
                for col in hdu.columns:
                    print(f"      {col.name:25s} format={col.format:8s} unit={col.unit or '-'}")

                # Print data shape and sample
                if hdu.data is not None and len(hdu.data) > 0:
                    print(f"\n    Rows: {len(hdu.data)}")
                    for col in hdu.columns:
                        data_col = hdu.data[col.name]
                        if hasattr(data_col, 'shape') and len(data_col.shape) > 1:
                            print(f"    {col.name}: shape={data_col.shape}, dtype={data_col.dtype}")
                        else:
                            arr = np.array(data_col, dtype=float)
                            nan_pct = np.isnan(arr).mean() * 100 if arr.dtype.kind == 'f' else 0
                            print(f"    {col.name}: min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}, "
                                  f"mean={np.nanmean(arr):.4f}, NaN={nan_pct:.1f}%")

        hdul.close()
    except Exception as e:
        print(f"  [ERROR] Could not read: {e}")

print("\n" + "=" * 80)
print("DONE — HEL1OS data inspection complete.")
