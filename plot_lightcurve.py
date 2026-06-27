"""
Plot SoLEXS lightcurve for ANY date.
Usage: python plot_lightcurve.py
  → Change ONLY the 'data_folder' variable below.
"""
from astropy.io import fits
from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob

# ╔═══════════════════════════════════════════════════════════╗
# ║  CHANGE THIS ONE LINE TO PLOT A DIFFERENT DATE           ║
# ╚═══════════════════════════════════════════════════════════╝
data_folder = r"dataset\AL1_SLX_L1_20260606_v1.0\AL1_SLX_L1_20260606_v1.0"

# ── Auto-detect files ────────────────────────────────────────
# PRADAN downloads sometimes have double-nested folders:
#   folder/SDD2/...          (flat)
#   folder/folder/SDD2/...   (nested)
# We handle both automatically.
search_roots = [data_folder]
# Check for a nested subfolder with the same name
for item in os.listdir(data_folder):
    nested = os.path.join(data_folder, item)
    if os.path.isdir(nested) and item.startswith("AL1_"):
        search_roots.insert(0, nested)  # prefer the nested one

lc_file = None
gti_file = None
sdd = None

for root in search_roots:
    for detector in ["SDD2", "SDD1"]:
        sdd_dir = os.path.join(root, detector)
        if not os.path.isdir(sdd_dir):
            continue
        lc_files = glob.glob(os.path.join(sdd_dir, "*.lc.gz")) + glob.glob(os.path.join(sdd_dir, "*.lc"))
        gti_files = glob.glob(os.path.join(sdd_dir, "*.gti.gz")) + glob.glob(os.path.join(sdd_dir, "*.gti"))
        if lc_files:
            lc_file = lc_files[0]
            gti_file = gti_files[0] if gti_files else None
            sdd = detector
            break
    if lc_file:
        break

if not lc_file:
    raise FileNotFoundError(f"No .lc files found in {data_folder} (checked flat and nested structures)")

print(f"Using detector: {sdd}")
print(f"Lightcurve: {lc_file}")
print(f"GTI: {gti_file}")

# ── Load the lightcurve ──────────────────────────────────────
hdul = fits.open(lc_file)
data = hdul[1].data
header = hdul[1].header

time_raw = data["TIME"]
counts = data["COUNTS"]

# Auto-detect date from FITS header (no hardcoding!)
mjd_ref = header["MJDREFI"] + header["MJDREFF"]
tstart_astropy = Time(mjd_ref + header["TSTART"] / 86400.0, format="mjd")
date_str = tstart_astropy.datetime.strftime("%Y-%m-%d")
print(f"Date: {date_str}")

# Convert all timestamps to Python datetime via astropy
times_utc = [Time(mjd_ref + t / 86400.0, format="mjd").datetime for t in time_raw]

# Handle NaN values
mask = ~np.isnan(counts)
times_clean = np.array(times_utc)[mask]
counts_clean = counts[mask]
hdul.close()

# ── Load GTI ─────────────────────────────────────────────────
if gti_file:
    hdul_gti = fits.open(gti_file)
    if len(hdul_gti) > 1 and len(hdul_gti[1].data) > 0:
        gti_data = hdul_gti[1].data
        print(f"Good Time Intervals: {len(gti_data)}")
        for i, row in enumerate(gti_data):
            dur = (row["STOP"] - row["START"]) / 3600
            print(f"  GTI[{i}]: duration = {dur:.1f} hours")
    hdul_gti.close()

# ── Basic statistics ─────────────────────────────────────────
print(f"\n=== {date_str} — SoLEXS {sdd} Lightcurve ===")
print(f"Total points: {len(counts_clean)}")
print(f"Counts/sec: min={counts_clean.min():.0f}, max={counts_clean.max():.0f}, "
      f"mean={counts_clean.mean():.1f}, median={np.median(counts_clean):.1f}")

median_bg = np.median(counts_clean)
threshold = 3 * median_bg
above_threshold = counts_clean > threshold
print(f"\nBackground (median): {median_bg:.1f} counts/s")
print(f"Detection threshold (3x median): {threshold:.1f} counts/s")
print(f"Points above threshold: {above_threshold.sum()} ({above_threshold.sum()/len(counts_clean)*100:.2f}%)")

peak_idx = np.argmax(counts_clean)
print(f"Peak: {counts_clean[peak_idx]:.0f} counts/s at {times_clean[peak_idx].strftime('%H:%M:%S')} UTC")

# ── PLOT ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Panel 1: Linear scale
ax1 = axes[0]
ax1.plot(times_clean, counts_clean, linewidth=0.3, color="#1976D2", alpha=0.8)
ax1.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label=f"3x median = {threshold:.0f}")
ax1.set_ylabel("Counts/s", fontsize=11)
ax1.set_title(f"SoLEXS {sdd} — {date_str} (Linear Scale)", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Log scale
ax2 = axes[1]
counts_pos = np.where(counts_clean > 0, counts_clean, 0.1)
ax2.semilogy(times_clean, counts_pos, linewidth=0.3, color="#E65100", alpha=0.8)
ax2.axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
ax2.set_ylabel("Counts/s (log)", fontsize=11)
ax2.set_title("Log Scale", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3, which="both")

# Panel 3: 1-minute binned
bin_size = 60
n_bins = len(counts_clean) // bin_size
if n_bins > 0:
    counts_binned = counts_clean[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
    times_binned = times_clean[:n_bins*bin_size:bin_size]
    ax3 = axes[2]
    ax3.plot(times_binned, counts_binned, linewidth=0.8, color="#2E7D32")
    ax3.axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
    peak_bin_idx = np.argmax(counts_binned)
    ax3.plot(times_binned[peak_bin_idx], counts_binned[peak_bin_idx], "rv", markersize=10, label="Peak")
    ax3.set_ylabel("Counts/s (1-min avg)", fontsize=11)
    ax3.set_title("1-Minute Binned", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

axes[2].set_xlabel("Time (UTC)", fontsize=11)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

plt.tight_layout()
out_name = f"lightcurve_{date_str}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_name}")
plt.show()
