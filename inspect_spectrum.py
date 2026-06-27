"""Inspect spectrum with NaN handling."""
from astropy.io import fits
import numpy as np
import os, glob

folder = "dataset/AL1_SLX_L1_20260611_v1.0"
for root in [folder, os.path.join(folder, os.path.basename(folder))]:
    pi_files = glob.glob(os.path.join(root, "SDD2", "*.pi.gz"))
    if pi_files:
        break

hdul = fits.open(pi_files[0])
counts = hdul[1].data["COUNTS"]  # shape: (86400, 340)

print(f"Shape: {counts.shape}")
print(f"NaN count: {np.isnan(counts).sum()} / {counts.size}")
print(f"NaN rows: {np.isnan(counts).any(axis=1).sum()} / {counts.shape[0]}")

# Use nansum instead
chan_sums = np.nansum(counts, axis=0)
print(f"\nChannel sums (nansum, total counts across 24h):")
for start in range(0, 100, 10):
    end = start + 10
    vals = chan_sums[start:end]
    print(f"  Ch {start:3d}-{end-1:3d}: {vals.astype(int)}")

print(f"\nPeak channel: {np.argmax(chan_sums)} ({chan_sums.max():.0f} total counts)")
print(f"Channels with >10000 counts: {(chan_sums > 10000).sum()}")
print(f"Channels with >1000 counts:  {(chan_sums > 1000).sum()}")
print(f"Channels with >100 counts:   {(chan_sums > 100).sum()}")

# Find a good flare second (not NaN, high counts)
total_per_sec = np.nansum(counts, axis=1)
# Skip NaN-heavy rows
valid_mask = ~np.isnan(counts).any(axis=1)
print(f"\nValid (no-NaN) rows: {valid_mask.sum()} / {counts.shape[0]}")

# Among valid rows, find quiet and peak
valid_totals = total_per_sec.copy()
valid_totals[~valid_mask] = np.nan

quiet_idx = int(np.nanargmin(valid_totals[1000:5000]) + 1000)
peak_idx = int(np.nanargmax(valid_totals))

print(f"\nQuiet second (idx={quiet_idx}, total={total_per_sec[quiet_idx]:.0f}):")
print(f"  Channels 10-39: {counts[quiet_idx][10:40].astype(int)}")

print(f"\nFlare peak second (idx={peak_idx}, total={total_per_sec[peak_idx]:.0f}):")
print(f"  Channels 10-39: {counts[peak_idx][10:40].astype(int)}")

# Compare quiet vs peak spectral shape
print(f"\nSpectral comparison (quiet vs peak):")
for band_name, ch_start, ch_end in [
    ("Band A (ch 10-19)", 10, 20),
    ("Band B (ch 20-34)", 20, 35),
    ("Band C (ch 35-49)", 35, 50),
    ("Band D (ch 50-79)", 50, 80),
    ("Band E (ch 80-119)", 80, 120),
    ("Band F (ch 120+)", 120, 340),
]:
    q = np.nansum(counts[quiet_idx][ch_start:ch_end])
    p = np.nansum(counts[peak_idx][ch_start:ch_end])
    ratio = p / max(q, 0.1)
    print(f"  {band_name}: quiet={q:.0f}, peak={p:.0f}, ratio={ratio:.1f}x")

# Create energy-band lightcurves for the flare period
# M-class flare on June 11 was around 08:26-08:38 UTC = seconds 30360-31080
flare_start = 30000
flare_end = 32000
print(f"\nEnergy-band lightcurves around M-class flare (seconds {flare_start}-{flare_end}):")

bands = {
    "soft": (10, 25),
    "medium": (25, 50),
    "hard": (50, 100),
    "vhard": (100, 200),
}

for bname, (ch_lo, ch_hi) in bands.items():
    band_lc = np.nansum(counts[flare_start:flare_end, ch_lo:ch_hi], axis=1)
    bg_lc = np.nansum(counts[flare_start:flare_start+300, ch_lo:ch_hi], axis=1)
    bg_level = np.nanmedian(bg_lc)
    peak_val = np.nanmax(band_lc)
    peak_offset = np.nanargmax(band_lc)
    print(f"  {bname:8s} (ch {ch_lo:3d}-{ch_hi-1:3d}): bg={bg_level:.1f}, peak={peak_val:.0f}, "
          f"ratio={peak_val/max(bg_level,0.1):.1f}x, peak_at=+{peak_offset}s")

hdul.close()
