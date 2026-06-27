"""
Solar Flare Forecasting Model v3 — Energy-Resolved
====================================================
Key improvements over v2:
  1. Multi-energy-band features from .pi spectrum files
  2. Spectral hardness ratio (hard/soft) — key precursor indicator
  3. Labels based on PEAK time (not start) for more lead time
  4. Longer history window (60 min)
  5. Lower default threshold for better recall
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import glob
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


# =================================================================
# CONFIGURATION
# =================================================================
DATASET_DIR = "dataset"
CATALOG_FILE = "master_flare_catalog.json"
HISTORY_WINDOW = 3600    # 60 min history (increased from 30)
STEP_SIZE = 60           # sample every 60s
HORIZON_SEC = 900        # predict 15 min ahead

# Energy bands (channel ranges in the 340-channel SoLEXS spectrum)
ENERGY_BANDS = {
    "soft":   (10, 25),    # Lowest energy — thermal, gradual
    "medium": (25, 50),    # Mid energy
    "hard":   (50, 100),   # High energy — non-thermal, impulsive
    "vhard":  (100, 200),  # Very high energy — flare-only signal
}


# =================================================================
# DATA LOADING
# =================================================================
def find_files(folder_path):
    """Find .lc and .pi files, handling nested PRADAN structure."""
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


def load_per_day():
    """Load lightcurve and energy-band data per day."""
    folders = sorted([f for f in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, f)) and f.startswith("AL1_")])
    days = []
    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        lc_path, pi_path, det = find_files(folder_path)
        if not lc_path:
            continue

        # Load lightcurve
        hdul = fits.open(lc_path)
        time_raw = hdul[1].data["TIME"].copy()
        counts_total = np.nan_to_num(hdul[1].data["COUNTS"].copy(), nan=0.0)
        header = hdul[1].header
        mjd_ref = header["MJDREFI"] + header["MJDREFF"]
        tstart = Time(mjd_ref + header["TSTART"] / 86400.0, format="mjd")
        date_str = tstart.datetime.strftime("%Y-%m-%d")
        hdul.close()

        # Load energy-band lightcurves from spectrum file
        band_data = {}
        if pi_path:
            try:
                hdul_pi = fits.open(pi_path)
                spec_counts = hdul_pi[1].data["COUNTS"]  # shape: (86400, 340)
                for band_name, (ch_lo, ch_hi) in ENERGY_BANDS.items():
                    band_lc = np.nansum(spec_counts[:, ch_lo:ch_hi], axis=1)
                    band_data[band_name] = band_lc.astype(np.float64)
                hdul_pi.close()
                print(f"  Loaded {date_str} ({det}): lc + spectrum (4 bands)")
            except Exception as e:
                print(f"  Loaded {date_str} ({det}): lc only (spectrum error: {e})")
        else:
            print(f"  Loaded {date_str} ({det}): lc only (no .pi file)")

        days.append({
            "date": date_str,
            "times": time_raw,
            "counts": counts_total,
            "bands": band_data,
            "has_spectrum": len(band_data) > 0,
        })
    return days


# =================================================================
# FEATURE ENGINEERING (v3 — with energy bands)
# =================================================================
def engineer_features_v3(times, counts, bands, has_spectrum):
    """
    Extract features including energy-resolved spectral features.
    
    NEW in v3:
    - Per-band statistics (soft, medium, hard, vhard)
    - Hardness ratios (hard/soft, vhard/medium)
    - Spectral slope changes
    - Band-specific derivatives
    """
    n = len(counts)
    
    # Pre-compute smoothed versions
    smooth_30s = uniform_filter1d(counts, size=30)
    smooth_60s = uniform_filter1d(counts, size=60)
    smooth_300s = uniform_filter1d(counts, size=300)
    deriv = np.gradient(smooth_30s)
    deriv2 = np.gradient(deriv)
    log_counts = np.log1p(counts)

    # Pre-compute band smoothing
    band_smooth = {}
    band_deriv = {}
    if has_spectrum:
        for bname, blc in bands.items():
            bs = uniform_filter1d(blc, size=60)
            band_smooth[bname] = bs
            band_deriv[bname] = np.gradient(bs)

    features_list = []
    sample_indices = []

    for i in range(HISTORY_WINDOW, n, STEP_SIZE):
        win = counts[i - HISTORY_WINDOW:i]
        win_smooth = smooth_30s[i - HISTORY_WINDOW:i]
        win_deriv = deriv[i - HISTORY_WINDOW:i]
        win_log = log_counts[i - HISTORY_WINDOW:i]
        recent_5min = counts[max(0, i - 300):i]
        recent_10min = counts[max(0, i - 600):i]
        earlier = counts[i - HISTORY_WINDOW:max(0, i - 600)]

        feat = {}

        # ── Total counts features (same as v2) ──
        feat["mean"] = np.mean(win)
        feat["std"] = np.std(win)
        feat["max"] = np.max(win)
        feat["median"] = np.median(win)
        m, s = np.mean(win), np.std(win)
        feat["skewness"] = np.mean(((win - m) / max(s, 0.01)) ** 3)
        feat["kurtosis"] = np.mean(((win - m) / max(s, 0.01)) ** 4) - 3
        feat["iqr"] = np.percentile(win, 75) - np.percentile(win, 25)
        feat["p90"] = np.percentile(win, 90)
        feat["p99"] = np.percentile(win, 99)
        feat["log_mean"] = np.mean(win_log)
        feat["log_std"] = np.std(win_log)
        feat["current"] = smooth_30s[i - 1]
        feat["current_raw"] = counts[i - 1]

        # Multi-scale ratios
        feat["ratio_60_300"] = smooth_60s[i-1] / max(smooth_300s[i-1], 0.1)

        # Derivatives
        feat["deriv_mean"] = np.mean(win_deriv)
        feat["deriv_max"] = np.max(win_deriv)
        feat["deriv_current"] = deriv[i - 1]
        recent_deriv = deriv[max(0, i - 300):i]
        feat["deriv_recent_mean"] = np.mean(recent_deriv)
        feat["deriv_recent_max"] = np.max(recent_deriv)

        # Trend
        feat["recent_vs_earlier"] = np.mean(recent_5min) / max(np.mean(earlier), 0.1) if len(earlier) > 0 else 1.0
        feat["recent10_vs_earlier"] = np.mean(recent_10min) / max(np.mean(earlier), 0.1) if len(earlier) > 0 else 1.0
        x = np.arange(len(win_smooth))
        coeffs = np.polyfit(x, win_smooth, 1)
        feat["trend_slope"] = coeffs[0]

        # Background
        bg = np.percentile(win, 30)
        feat["background"] = bg
        feat["current_over_bg"] = smooth_30s[i-1] / max(bg, 0.1)

        # Monotonic rise
        n_seg = 6
        seg_len = len(win) // n_seg
        if seg_len > 0:
            seg_means = [np.mean(win[j*seg_len:(j+1)*seg_len]) for j in range(n_seg)]
            rising = sum(1 for j in range(1, n_seg) if seg_means[j] > seg_means[j-1])
            feat["monotonic_rise"] = rising / (n_seg - 1)
            feat["seg_ratio"] = seg_means[-1] / max(seg_means[0], 0.1)
        else:
            feat["monotonic_rise"] = 0
            feat["seg_ratio"] = 1

        # Variability
        feat["cv"] = np.std(win) / max(np.mean(win), 0.1)

        # ── NEW: Energy-band features ──
        if has_spectrum:
            for bname in ENERGY_BANDS:
                blc = bands[bname]
                bwin = blc[i - HISTORY_WINDOW:i]
                bs = band_smooth[bname]
                bd = band_deriv[bname]

                # Band statistics
                feat[f"{bname}_mean"] = np.mean(bwin)
                feat[f"{bname}_max"] = np.max(bwin)
                feat[f"{bname}_current"] = bs[i - 1]
                feat[f"{bname}_std"] = np.std(bwin)

                # Band derivatives
                bd_win = bd[i - HISTORY_WINDOW:i]
                feat[f"{bname}_deriv_mean"] = np.mean(bd_win)
                feat[f"{bname}_deriv_max"] = np.max(bd_win)
                feat[f"{bname}_deriv_current"] = bd[i - 1]

                # Band recent trend
                b_recent = blc[max(0, i - 300):i]
                b_earlier = blc[i - HISTORY_WINDOW:max(0, i - 600)]
                feat[f"{bname}_recent_trend"] = np.mean(b_recent) / max(np.mean(b_earlier), 0.01) if len(b_earlier) > 0 else 1.0

                # Band background ratio
                b_bg = np.percentile(bwin, 30)
                feat[f"{bname}_over_bg"] = bs[i-1] / max(b_bg, 0.01)

                # Band monotonic rise
                if seg_len > 0:
                    b_segs = [np.mean(bwin[j*seg_len:(j+1)*seg_len]) for j in range(n_seg)]
                    b_rising = sum(1 for j in range(1, n_seg) if b_segs[j] > b_segs[j-1])
                    feat[f"{bname}_mono_rise"] = b_rising / (n_seg - 1)

            # ── CRITICAL: Hardness ratios ──
            # These detect spectral hardening BEFORE a flare peaks
            soft_val = max(band_smooth["soft"][i-1], 0.01)
            med_val = max(band_smooth["medium"][i-1], 0.01)
            hard_val = band_smooth["hard"][i-1]
            vhard_val = band_smooth["vhard"][i-1]

            feat["hardness_hard_soft"] = hard_val / soft_val
            feat["hardness_vhard_soft"] = vhard_val / soft_val
            feat["hardness_hard_med"] = hard_val / med_val
            feat["hardness_vhard_med"] = vhard_val / med_val

            # Hardness ratio TREND (is the spectrum hardening?)
            hr_window = bands["hard"][max(0, i-300):i] / np.maximum(bands["soft"][max(0, i-300):i], 0.01)
            hr_earlier = bands["hard"][i-HISTORY_WINDOW:max(0, i-600)] / np.maximum(bands["soft"][i-HISTORY_WINDOW:max(0, i-600)], 0.01)
            feat["hardness_trend"] = np.mean(hr_window) / max(np.mean(hr_earlier), 0.001) if len(hr_earlier) > 0 else 1.0

            # Hardness derivative
            hr_full = bands["hard"][i-HISTORY_WINDOW:i] / np.maximum(bands["soft"][i-HISTORY_WINDOW:i], 0.01)
            hr_smooth = uniform_filter1d(hr_full, size=60)
            feat["hardness_deriv"] = np.gradient(hr_smooth)[-1]
            feat["hardness_deriv_max"] = np.max(np.gradient(hr_smooth)[-300:])

            # Hard band activity (any hard X-rays = likely pre-flare)
            feat["hard_active"] = float(np.mean(bands["hard"][max(0, i-300):i]) > 1.0)
            feat["vhard_active"] = float(np.mean(bands["vhard"][max(0, i-60):i]) > 0.5)

        features_list.append(feat)
        sample_indices.append(i)

    df = pd.DataFrame(features_list)
    return df.values, list(df.columns), np.array(sample_indices)


def create_labels(times, catalog, horizon_sec):
    """Labels based on flare PEAK time (gives more lead time than start)."""
    labels = np.zeros(len(times), dtype=np.int32)
    for flare in catalog:
        peak_t = flare["peak_time"]
        mask = (times >= peak_t - horizon_sec) & (times < peak_t)
        labels[mask] = 1
    return labels


def find_optimal_threshold(y_true, y_prob):
    """Find threshold maximizing F1, biased toward recall."""
    best_score = 0
    best_thresh = 0.3
    for thresh in np.arange(0.05, 0.90, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        # Use F-beta with beta=2 (recall twice as important as precision)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        beta = 2
        fbeta = (1 + beta**2) * prec * rec / max((beta**2 * prec + rec), 1e-8)
        if fbeta > best_score:
            best_score = fbeta
            best_thresh = thresh
    return best_thresh, best_score


# =================================================================
# MAIN — LEAVE-ONE-DAY-OUT CV
# =================================================================
def main():
    print("[*] Solar Flare Forecasting v3 (Energy-Resolved)")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data with spectrum...")
    days = load_per_day()

    # 2. Load catalog
    with open(CATALOG_FILE, 'r') as f:
        catalog = json.load(f)["catalog"]
    print(f"\n  {len(catalog)} flares in catalog")

    # 3. Feature engineering
    print("\n[2] Engineering features (60-min window + energy bands)...")
    day_features = []
    for d in days:
        print(f"  {d['date']}...", end=" ")
        X, feat_names, idx = engineer_features_v3(
            d["times"], d["counts"], d["bands"], d["has_spectrum"]
        )
        sample_times = d["times"][idx]
        y = create_labels(d["times"], catalog, HORIZON_SEC)[idx]
        day_flares = [f for f in catalog if f["date"] == d["date"]]
        print(f"-> {len(X)} samples, {X.shape[1]} features, {y.sum()} pos, {len(day_flares)} flares")
        day_features.append({
            "date": d["date"], "X": X, "y": y,
            "times": sample_times, "flares": day_flares,
        })
    feature_names = feat_names
    n_days = len(day_features)

    # 4. Leave-one-day-out CV
    print(f"\n[3] Leave-One-Day-Out Cross-Validation ({n_days} folds)")
    print("=" * 60)

    all_lead_times = []
    all_y_true = []
    all_y_prob = []
    all_y_pred = []

    for test_idx in range(n_days):
        test_day = day_features[test_idx]
        train_days = [day_features[i] for i in range(n_days) if i != test_idx]

        X_train = np.vstack([d["X"] for d in train_days])
        y_train = np.concatenate([d["y"] for d in train_days])
        X_test = test_day["X"]
        y_test = test_day["y"]
        times_test = test_day["times"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            scale_pos_weight=n_neg / max(n_pos, 1) * 0.7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.05,
            reg_lambda=1.0,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train_s, y_train, verbose=False)

        y_prob = model.predict_proba(X_test_s)[:, 1]

        # Find optimal threshold (biased toward recall)
        y_train_prob = model.predict_proba(X_train_s)[:, 1]
        opt_thresh, opt_score = find_optimal_threshold(y_train, y_train_prob)
        y_pred = (y_prob >= opt_thresh).astype(int)

        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        tpr = tp / max(tp + fn, 1)

        print(f"\n  Test: {test_day['date']} | {len(test_day['flares'])} flares | "
              f"thresh={opt_thresh:.2f} | TP={tp} FP={fp} FN={fn} TPR={tpr:.3f}")

        for flare in test_day["flares"]:
            peak_t = flare["peak_time"]
            fid = flare.get("flare_id", "?")
            fclass = flare.get("flare_class", "?")

            pre_mask = (times_test >= peak_t - 1800) & (times_test < peak_t)
            if pre_mask.sum() == 0:
                continue

            pre_probs = y_prob[pre_mask]
            pre_times = times_test[pre_mask]
            alert_mask = pre_probs >= opt_thresh

            if alert_mask.sum() > 0:
                first_alert = pre_times[np.where(alert_mask)[0][0]]
                lead_min = (peak_t - first_alert) / 60.0
                all_lead_times.append({"flare_id": fid, "class": fclass,
                                       "lead_time_min": lead_min, "date": test_day["date"]})
                print(f"    {fid} ({fclass}): Alert {lead_min:.1f} min before PEAK")
            else:
                max_prob = pre_probs.max()
                all_lead_times.append({"flare_id": fid, "class": fclass,
                                       "lead_time_min": 0, "date": test_day["date"]})
                print(f"    {fid} ({fclass}): MISSED (max prob={max_prob:.3f})")

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

    # 5. Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (v3 - Energy-Resolved)")
    print("=" * 60)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=["No Flare", "Flare"], zero_division=0))

    roc_auc = roc_auc_score(all_y_true, all_y_prob) if all_y_true.sum() > 0 else 0
    pr_auc = average_precision_score(all_y_true, all_y_prob) if all_y_true.sum() > 0 else 0
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    print(f"\nOverall: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"True Positive Rate:  {tpr:.4f}")
    print(f"False Alarm Rate:    {far:.4f}")

    # Lead times
    print("\n--- Lead Time Summary ---")
    detected = [lt for lt in all_lead_times if lt["lead_time_min"] > 0]
    missed = [lt for lt in all_lead_times if lt["lead_time_min"] == 0]
    print(f"Flares detected: {len(detected)}/{len(all_lead_times)}")
    if detected:
        lead_vals = [lt["lead_time_min"] for lt in detected]
        print(f"Average lead time: {np.mean(lead_vals):.1f} min")
        print(f"Median lead time:  {np.median(lead_vals):.1f} min")
        print(f"Max lead time:     {np.max(lead_vals):.1f} min")

    print("\nPer-flare results:")
    for lt in all_lead_times:
        status = f"{lt['lead_time_min']:.1f} min before peak" if lt['lead_time_min'] > 0 else "MISSED"
        print(f"  {lt['flare_id']} ({lt['class']}, {lt['date']}): {status}")

    # 6. Train final model
    print("\n[4] Training final model on all data...")
    X_all = np.vstack([d["X"] for d in day_features])
    y_all = np.concatenate([d["y"] for d in day_features])
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X_all)

    final_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        scale_pos_weight=(len(y_all) - y_all.sum()) / max(y_all.sum(), 1) * 0.7,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=1.0, eval_metric="aucpr",
        random_state=42, verbosity=0,
    )
    final_model.fit(X_all_s, y_all, verbose=False)

    y_all_prob = final_model.predict_proba(X_all_s)[:, 1]
    final_thresh, _ = find_optimal_threshold(y_all, y_all_prob)

    with open("forecast_model_v3.pkl", "wb") as f:
        pickle.dump({
            "model": final_model, "scaler": scaler_final,
            "feature_names": feature_names, "threshold": final_thresh,
            "horizon_sec": HORIZON_SEC, "history_window": HISTORY_WINDOW,
            "energy_bands": ENERGY_BANDS,
        }, f)
    print(f"  Saved: forecast_model_v3.pkl (threshold={final_thresh:.2f})")

    # 7. Plots
    print("\n[5] Generating plots...")

    # Feature importance
    importance = final_model.feature_importances_
    top_n = min(25, len(feature_names))
    indices = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = []
    for idx in indices:
        name = feature_names[idx]
        if "hard" in name.lower():
            colors.append("#F44336")
        elif "soft" in name.lower():
            colors.append("#2196F3")
        elif "medium" in name.lower():
            colors.append("#FF9800")
        else:
            colors.append("#607D8B")
    ax.barh(range(len(indices)), importance[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top Features (red=hard, blue=soft, orange=medium, gray=total)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("feature_importance_v3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: feature_importance_v3.png")

    # Lead time chart
    if all_lead_times:
        fig, ax = plt.subplots(figsize=(14, 5))
        ids = [lt["flare_id"] for lt in all_lead_times]
        vals = [lt["lead_time_min"] for lt in all_lead_times]
        colors_lt = ["#4CAF50" if v > 0 else "#F44336" for v in vals]
        ax.bar(range(len(ids)), vals, color=colors_lt)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([f"{lt['flare_id']}\n({lt['class']})" for lt in all_lead_times],
                          fontsize=5, rotation=45, ha="right")
        ax.set_ylabel("Lead Time (min before peak)")
        ax.set_title("Forecast Lead Time per Flare (v3 - Energy-Resolved)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("lead_times_v3.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: lead_times_v3.png")

    # ROC + PR
    if roc_auc > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fpr_arr, tpr_arr, _ = roc_curve(all_y_true, all_y_prob)
        ax1.plot(fpr_arr, tpr_arr, color="#1976D2", linewidth=2, label=f"AUC={roc_auc:.3f}")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.set_title("ROC (v3)", fontweight="bold"); ax1.legend(); ax1.grid(True, alpha=0.3)
        prec, rec, _ = precision_recall_curve(all_y_true, all_y_prob)
        ax2.plot(rec, prec, color="#E65100", linewidth=2, label=f"AP={pr_auc:.3f}")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title("PR Curve (v3)", fontweight="bold"); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("eval_curves_v3.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: eval_curves_v3.png")

    # Save results
    results = {
        "version": "v3", "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tpr": tpr, "far": far, "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn), "lead_times": all_lead_times,
        "n_detected": len(detected), "n_total": len(all_lead_times),
        "threshold": final_thresh, "n_features": len(feature_names),
    }
    with open("forecast_results_v3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("[OK] Forecasting v3 complete!")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  Flares detected: {len(detected)}/{len(all_lead_times)}")
    if detected:
        print(f"  Avg lead time: {np.mean(lead_vals):.1f} min")
        print(f"  Max lead time: {np.max(lead_vals):.1f} min")
    print(f"  Features: {len(feature_names)} (including {sum(1 for f in feature_names if any(b in f for b in ENERGY_BANDS))} spectral)")
    print("=" * 60)


if __name__ == "__main__":
    main()
