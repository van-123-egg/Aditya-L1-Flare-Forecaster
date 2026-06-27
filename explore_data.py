"""Quick exploration script for SoLEXS Level-1 FITS data."""
from astropy.io import fits
from astropy.time import Time
import numpy as np

# --- 1. Inspect the Lightcurve (.lc) ---
lc_file = r"dataset\AL1_SLX_L1_20260613_v1.0\SDD2\AL1_SOLEXS_20260613_SDD2_L1.lc.gz"
hdul_lc = fits.open(lc_file)

data = hdul_lc[1].data
h = hdul_lc[1].header

print("=" * 60)
print("LIGHTCURVE FILE (SDD2 - .lc)")
print("=" * 60)
print(f"Rows: {len(data)}")
print(f"Columns: TIME, COUNTS")
print(f"Cadence (TIMEDEL): {h['TIMEDEL']} seconds")
print(f"TSTART: {h['TSTART']}, TSTOP: {h['TSTOP']}")
print(f"Duration: {(h['TSTOP'] - h['TSTART']) / 3600:.1f} hours")
print(f"FILTER: {h['FILTER']}")

# Convert to UTC
mjd_ref = h["MJDREFI"] + h["MJDREFF"]
tstart_utc = Time(mjd_ref + h["TSTART"] / 86400.0, format="mjd")
tstop_utc = Time(mjd_ref + h["TSTOP"] / 86400.0, format="mjd")
print(f"Start UTC: {tstart_utc.iso}")
print(f"Stop  UTC: {tstop_utc.iso}")

time_col = data["TIME"]
counts_col = data["COUNTS"]
print(f"\nTIME range: {time_col[0]} to {time_col[-1]}")
print(f"COUNTS: min={np.nanmin(counts_col):.2f}, max={np.nanmax(counts_col):.2f}, mean={np.nanmean(counts_col):.2f}")
nan_count = np.isnan(counts_col).sum()
print(f"NaN values: {nan_count} / {len(counts_col)}")
print(f"\nFirst 5 rows:")
for i in range(5):
    print(f"  TIME={time_col[i]:.1f}, COUNTS={counts_col[i]:.2f}")

hdul_lc.close()

# --- 2. Inspect the GTI (.gti) ---
print("\n" + "=" * 60)
print("GOOD TIME INTERVALS (SDD2 - .gti)")
print("=" * 60)
gti_file = r"dataset\AL1_SLX_L1_20260613_v1.0\SDD2\AL1_SOLEXS_20260613_SDD2_L1.gti.gz"
hdul_gti = fits.open(gti_file)
hdul_gti.info()
if len(hdul_gti) > 1:
    gti_data = hdul_gti[1].data
    print(f"GTI columns: {hdul_gti[1].columns.names}")
    print(f"Number of GTI intervals: {len(gti_data)}")
    for i in range(min(5, len(gti_data))):
        print(f"  GTI[{i}]: START={gti_data[i][0]:.1f}, STOP={gti_data[i][1]:.1f}")
hdul_gti.close()

# --- 3. Inspect the Spectrum (.pi) ---
print("\n" + "=" * 60)
print("SPECTRUM FILE (SDD2 - .pi)")
print("=" * 60)
pi_file = r"dataset\AL1_SLX_L1_20260613_v1.0\SDD2\AL1_SOLEXS_20260613_SDD2_L1.pi.gz"
hdul_pi = fits.open(pi_file)
hdul_pi.info()
if len(hdul_pi) > 1:
    print(f"PI columns: {hdul_pi[1].columns.names}")
    print(f"PI rows: {len(hdul_pi[1].data)}")
    pi_h = hdul_pi[1].header
    for key in ["EXTNAME", "CONTENT", "FILTER", "TIMEDEL", "TSTART", "TSTOP"]:
        if key in pi_h:
            print(f"  {key} = {pi_h[key]}")
    # Show shape of data
    for col_name in hdul_pi[1].columns.names[:5]:
        col_data = hdul_pi[1].data[col_name]
        if hasattr(col_data, "shape"):
            print(f"  Column '{col_name}': shape={col_data.shape}, dtype={col_data.dtype}")
hdul_pi.close()

# --- 4. Also check SDD1 ---
print("\n" + "=" * 60)
print("SDD1 FILES")
print("=" * 60)
import os
sdd1_dir = r"dataset\AL1_SLX_L1_20260613_v1.0\SDD1"
for f in os.listdir(sdd1_dir):
    fpath = os.path.join(sdd1_dir, f)
    print(f"  {f} ({os.path.getsize(fpath)/1024:.1f} KB)")
    if f.endswith(".gti") or f.endswith(".gti.gz"):
        try:
            hdul = fits.open(fpath)
            hdul.info()
            if len(hdul) > 1:
                print(f"    Columns: {hdul[1].columns.names}")
                print(f"    Rows: {len(hdul[1].data)}")
            hdul.close()
        except Exception as e:
            print(f"    Error: {e}")
