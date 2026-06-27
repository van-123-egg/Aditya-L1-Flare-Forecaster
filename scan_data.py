"""Scan all dataset folders for data availability."""
import os, glob
from astropy.io import fits
import numpy as np

for folder in sorted(os.listdir("dataset")):
    path = os.path.join("dataset", folder)
    if not os.path.isdir(path):
        continue
    
    lc_found = False
    for root in [path, os.path.join(path, folder)]:
        for det in ["SDD2", "SDD1"]:
            sdd = os.path.join(root, det)
            if not os.path.isdir(sdd):
                continue
            lcs = glob.glob(os.path.join(sdd, "*.lc.gz")) + glob.glob(os.path.join(sdd, "*.lc"))
            pis = glob.glob(os.path.join(sdd, "*.pi.gz")) + glob.glob(os.path.join(sdd, "*.pi"))
            if lcs:
                try:
                    h = fits.open(lcs[0])
                    c = np.nan_to_num(h[1].data["COUNTS"], nan=0)
                    has_pi = "YES" if pis else "NO"
                    peak_ratio = c.max() / max(np.median(c), 1)
                    print(f"{folder} | {det} | pts={len(c)} | pi={has_pi} | "
                          f"max={c.max():.0f} mean={c.mean():.1f} median={np.median(c):.0f} | "
                          f"peak/med={peak_ratio:.1f}x")
                    h.close()
                    lc_found = True
                except Exception as e:
                    print(f"{folder} | {det} | ERROR: {e}")
                break
        if lc_found:
            break
    if not lc_found:
        print(f"{folder} | NO DATA FOUND")
