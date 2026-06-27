"""
Solar Flare Forecasting Model — v2 (Improved)
==============================================
Fixes from v1:
  1. Leave-one-day-out cross-validation (every flare gets tested)
  2. Optimal threshold tuning from validation set
  3. Better lead time analysis across all days
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
HISTORY_WINDOW = 1800    # 30 min of history
STEP_SIZE = 60           # sample every 60s
HORIZON_SEC = 900        # predict 15 min ahead


# =================================================================
# DATA LOADING
# =================================================================
def find_lc_file(folder_path):
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
            if lc_files:
                return lc_files[0], det
    return None, None


def load_per_day():
    """Load data separately per day for cross-validation."""
    folders = sorted([f for f in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, f)) and f.startswith("AL1_")])
    days = []
    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        lc_path, det = find_lc_file(folder_path)
        if not lc_path:
            continue
        hdul = fits.open(lc_path)
        data = hdul[1].data
        header = hdul[1].header
        time_raw = data["TIME"].copy()
        counts = np.nan_to_num(data["COUNTS"].copy(), nan=0.0)
        mjd_ref = header["MJDREFI"] + header["MJDREFF"]
        tstart = Time(mjd_ref + header["TSTART"] / 86400.0, format="mjd")
        date_str = tstart.datetime.strftime("%Y-%m-%d")
        hdul.close()
        days.append({"date": date_str, "times": time_raw, "counts": counts})
        print(f"  Loaded {date_str}: {len(counts)} points")
    return days


# =================================================================
# FEATURE ENGINEERING (same as v1)
# =================================================================
def engineer_features_for_day(times, counts):
    n = len(counts)
    smooth_10s = uniform_filter1d(counts, size=10)
    smooth_30s = uniform_filter1d(counts, size=30)
    smooth_60s = uniform_filter1d(counts, size=60)
    smooth_300s = uniform_filter1d(counts, size=300)
    deriv = np.gradient(smooth_30s)
    deriv2 = np.gradient(deriv)
    log_counts = np.log1p(counts)

    features_list = []
    sample_indices = []

    for i in range(HISTORY_WINDOW, n, STEP_SIZE):
        win = counts[i - HISTORY_WINDOW:i]
        win_smooth = smooth_30s[i - HISTORY_WINDOW:i]
        win_deriv = deriv[i - HISTORY_WINDOW:i]
        win_deriv2 = deriv2[i - HISTORY_WINDOW:i]
        win_log = log_counts[i - HISTORY_WINDOW:i]
        recent_5min = counts[max(0, i - 300):i]
        recent_1min = counts[max(0, i - 60):i]
        earlier = counts[i - HISTORY_WINDOW:max(0, i - 300)]

        feat = {}

        # Basic stats
        feat["mean"] = np.mean(win)
        feat["std"] = np.std(win)
        feat["min"] = np.min(win)
        feat["max"] = np.max(win)
        feat["median"] = np.median(win)
        m, s = np.mean(win), np.std(win)
        feat["skewness"] = np.mean(((win - m) / max(s, 0.01)) ** 3)
        feat["kurtosis"] = np.mean(((win - m) / max(s, 0.01)) ** 4) - 3
        feat["iqr"] = np.percentile(win, 75) - np.percentile(win, 25)
        feat["p90"] = np.percentile(win, 90)
        feat["p99"] = np.percentile(win, 99)

        # Log stats
        feat["log_mean"] = np.mean(win_log)
        feat["log_std"] = np.std(win_log)
        feat["log_max"] = np.max(win_log)

        # Current value
        feat["current"] = smooth_30s[i - 1]
        feat["current_raw"] = counts[i - 1]
        feat["current_log"] = log_counts[i - 1]

        # Multi-scale ratios
        feat["ratio_10_300"] = smooth_10s[i-1] / max(smooth_300s[i-1], 0.1)
        feat["ratio_30_300"] = smooth_30s[i-1] / max(smooth_300s[i-1], 0.1)
        feat["ratio_60_300"] = smooth_60s[i-1] / max(smooth_300s[i-1], 0.1)

        # Derivatives
        feat["deriv_mean"] = np.mean(win_deriv)
        feat["deriv_max"] = np.max(win_deriv)
        feat["deriv_min"] = np.min(win_deriv)
        feat["deriv_std"] = np.std(win_deriv)
        feat["deriv_current"] = deriv[i - 1]
        recent_deriv = deriv[max(0, i - 300):i]
        feat["deriv_recent_mean"] = np.mean(recent_deriv)
        feat["deriv_recent_max"] = np.max(recent_deriv)

        # Acceleration
        feat["accel_mean"] = np.mean(win_deriv2)
        feat["accel_max"] = np.max(win_deriv2)
        feat["accel_current"] = deriv2[i - 1]

        # Trend
        feat["recent_vs_earlier_mean"] = np.mean(recent_5min) / max(np.mean(earlier), 0.1) if len(earlier) > 0 else 1.0
        feat["recent_vs_earlier_max"] = np.max(recent_5min) / max(np.max(earlier), 0.1) if len(earlier) > 0 else 1.0
        feat["last_1min_vs_mean"] = np.mean(recent_1min) / max(np.mean(win), 0.1)
        x = np.arange(len(win_smooth))
        coeffs = np.polyfit(x, win_smooth, 1)
        feat["trend_slope"] = coeffs[0]
        feat["trend_intercept"] = coeffs[1]

        # Background
        bg = np.percentile(win, 30)
        feat["background"] = bg
        feat["current_over_bg"] = smooth_30s[i-1] / max(bg, 0.1)
        feat["max_over_bg"] = np.max(win) / max(bg, 0.1)

        # Pre-flare indicators
        n_segments = 6
        seg_len = len(win) // n_segments
        if seg_len > 0:
            seg_means = [np.mean(win[j*seg_len:(j+1)*seg_len]) for j in range(n_segments)]
            rising_count = sum(1 for j in range(1, n_segments) if seg_means[j] > seg_means[j-1])
            feat["monotonic_rise_score"] = rising_count / (n_segments - 1)
            feat["seg_ratio_last_first"] = seg_means[-1] / max(seg_means[0], 0.1)
        else:
            feat["monotonic_rise_score"] = 0
            feat["seg_ratio_last_first"] = 1

        # Variability
        sign_changes = np.diff(np.sign(win_deriv))
        feat["zero_crossings"] = np.count_nonzero(sign_changes)
        feat["cv"] = np.std(win) / max(np.mean(win), 0.1)

        # Peaks
        peaks, props = find_peaks(win_smooth, height=bg * 2, distance=30)
        feat["n_peaks"] = len(peaks)
        feat["max_peak_height"] = max(props["peak_heights"]) if len(peaks) > 0 else 0

        # Time features
        sod = times[i] - (times[i] // 86400) * 86400
        feat["hour_sin"] = np.sin(2 * np.pi * sod / 86400)
        feat["hour_cos"] = np.cos(2 * np.pi * sod / 86400)

        features_list.append(feat)
        sample_indices.append(i)

    df = pd.DataFrame(features_list)
    return df.values, list(df.columns), np.array(sample_indices)


def create_labels(times, catalog, horizon_sec):
    labels = np.zeros(len(times), dtype=np.int32)
    for flare in catalog:
        start_t = flare["start_time"]
        mask = (times >= start_t - horizon_sec) & (times < start_t)
        labels[mask] = 1
    return labels


# =================================================================
# FIND OPTIMAL THRESHOLD
# =================================================================
def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# =================================================================
# LEAVE-ONE-DAY-OUT CROSS-VALIDATION
# =================================================================
def main():
    print("[*] Solar Flare Forecasting Model v2 (Leave-One-Day-Out)")
    print("=" * 60)

    # 1. Load data per day
    print("\n[1] Loading data per day...")
    days = load_per_day()
    n_days = len(days)

    # 2. Load catalog
    print("\n[2] Loading flare catalog...")
    with open(CATALOG_FILE, 'r') as f:
        catalog = json.load(f)["catalog"]
    print(f"  {len(catalog)} flares total")

    # 3. Feature engineering per day
    print("\n[3] Engineering features per day...")
    day_features = []
    for d in days:
        print(f"  {d['date']}...", end=" ")
        X, feat_names, idx = engineer_features_for_day(d["times"], d["counts"])
        sample_times = d["times"][idx]
        y = create_labels(d["times"], catalog, HORIZON_SEC)[idx]

        # Count flares in this day
        day_flares = [f for f in catalog if f["date"] == d["date"]]
        print(f"-> {len(X)} samples, {y.sum()} positive, {len(day_flares)} flares")

        day_features.append({
            "date": d["date"],
            "X": X,
            "y": y,
            "times": sample_times,
            "flares": day_flares,
        })
    feature_names = feat_names

    # 4. Leave-one-day-out cross-validation
    print("\n[4] Leave-One-Day-Out Cross-Validation")
    print("=" * 60)

    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    all_times = []
    all_lead_times = []
    all_dates = []

    for test_idx in range(n_days):
        test_day = day_features[test_idx]
        train_days = [day_features[i] for i in range(n_days) if i != test_idx]

        # Assemble train set
        X_train = np.vstack([d["X"] for d in train_days])
        y_train = np.concatenate([d["y"] for d in train_days])
        X_test = test_day["X"]
        y_test = test_day["y"]
        times_test = test_day["times"]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1) * 0.5

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train_s, y_train, verbose=False)

        # Predict
        y_prob = model.predict_proba(X_test_s)[:, 1]

        # Find optimal threshold on this fold's train data
        y_train_prob = model.predict_proba(X_train_s)[:, 1]
        opt_thresh, opt_f1 = find_optimal_threshold(y_train, y_train_prob)

        y_pred = (y_prob >= opt_thresh).astype(int)

        # Metrics for this fold
        n_flares = len(test_day["flares"])
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()

        tpr = tp / max(tp + fn, 1)
        far = fp / max(fp + tn, 1)

        print(f"\n  Test: {test_day['date']} | {n_flares} flares | thresh={opt_thresh:.2f}")
        print(f"    TP={tp} FP={fp} FN={fn} TN={tn} | TPR={tpr:.3f} FAR={far:.4f}")

        # Lead time for each flare on this test day
        for flare in test_day["flares"]:
            flare_start = flare["start_time"]
            fid = flare.get("flare_id", "?")
            fclass = flare.get("flare_class", "?")

            pre_mask = (times_test >= flare_start - 1800) & (times_test < flare_start)
            if pre_mask.sum() == 0:
                print(f"    {fid} ({fclass}): No pre-flare data")
                continue

            pre_probs = y_prob[pre_mask]
            pre_times = times_test[pre_mask]
            alert_mask = pre_probs >= opt_thresh

            if alert_mask.sum() > 0:
                first_alert = pre_times[np.where(alert_mask)[0][0]]
                lead_min = (flare_start - first_alert) / 60.0
                all_lead_times.append({"flare_id": fid, "class": fclass,
                                       "lead_time_min": lead_min, "date": test_day["date"]})
                print(f"    {fid} ({fclass}): Alert {lead_min:.1f} min before start")
            else:
                max_prob = pre_probs.max()
                all_lead_times.append({"flare_id": fid, "class": fclass,
                                       "lead_time_min": 0, "date": test_day["date"]})
                print(f"    {fid} ({fclass}): MISSED (max prob={max_prob:.3f}, thresh={opt_thresh:.2f})")

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)
        all_times.extend(times_test)
        all_dates.extend([test_day["date"]] * len(y_test))

    # 5. Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (all folds)")
    print("=" * 60)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=["No Flare", "Flare"], zero_division=0))

    if all_y_true.sum() > 0 and (1 - all_y_true).sum() > 0:
        roc_auc = roc_auc_score(all_y_true, all_y_prob)
        pr_auc = average_precision_score(all_y_true, all_y_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")
    else:
        roc_auc, pr_auc = 0, 0

    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / max(tp + fn, 1)
    far = fp / max(fp + tn, 1)
    print(f"\nOverall: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"True Positive Rate:  {tpr:.4f}")
    print(f"False Alarm Rate:    {far:.4f}")

    # Lead time summary
    print("\n--- Lead Time Summary ---")
    detected = [lt for lt in all_lead_times if lt["lead_time_min"] > 0]
    missed = [lt for lt in all_lead_times if lt["lead_time_min"] == 0]
    print(f"Flares detected: {len(detected)}/{len(all_lead_times)}")
    print(f"Flares missed:   {len(missed)}/{len(all_lead_times)}")
    if detected:
        lead_vals = [lt["lead_time_min"] for lt in detected]
        print(f"Average lead time: {np.mean(lead_vals):.1f} min")
        print(f"Median lead time:  {np.median(lead_vals):.1f} min")
        print(f"Min lead time:     {np.min(lead_vals):.1f} min")
        print(f"Max lead time:     {np.max(lead_vals):.1f} min")

    for lt in all_lead_times:
        status = f"{lt['lead_time_min']:.1f} min" if lt['lead_time_min'] > 0 else "MISSED"
        print(f"  {lt['flare_id']} ({lt['class']}-class, {lt['date']}): {status}")

    # 6. Train final model on ALL data for deployment
    print("\n[6] Training final model on all data...")
    X_all = np.vstack([d["X"] for d in day_features])
    y_all = np.concatenate([d["y"] for d in day_features])
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X_all)

    n_pos = y_all.sum()
    n_neg = len(y_all) - n_pos
    final_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=n_neg / max(n_pos, 1) * 0.5,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="aucpr",
        random_state=42, verbosity=0,
    )
    final_model.fit(X_all_s, y_all, verbose=False)

    # Optimal threshold from all data
    y_all_prob = final_model.predict_proba(X_all_s)[:, 1]
    final_thresh, final_f1 = find_optimal_threshold(y_all, y_all_prob)
    print(f"  Optimal threshold: {final_thresh:.2f} (F1={final_f1:.3f})")

    # Save
    with open("forecast_model_v2.pkl", "wb") as f:
        pickle.dump({
            "model": final_model,
            "scaler": scaler_final,
            "feature_names": feature_names,
            "threshold": final_thresh,
            "horizon_sec": HORIZON_SEC,
            "history_window": HISTORY_WINDOW,
        }, f)
    print("  Saved: forecast_model_v2.pkl")

    # 7. Plots
    print("\n[7] Generating plots...")

    # Feature importance
    importance = final_model.feature_importances_
    indices = np.argsort(importance)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importance[indices], color="#1976D2")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title("Top 20 Features - 15min Forecast (v2)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("feature_importance_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: feature_importance_v2.png")

    # ROC and PR curves
    if roc_auc > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fpr_arr, tpr_arr, _ = roc_curve(all_y_true, all_y_prob)
        ax1.plot(fpr_arr, tpr_arr, color="#1976D2", linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve (Leave-One-Day-Out)", fontweight="bold")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        prec, rec, _ = precision_recall_curve(all_y_true, all_y_prob)
        ax2.plot(rec, prec, color="#E65100", linewidth=2, label=f"AP = {pr_auc:.3f}")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve", fontweight="bold")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("eval_curves_v2.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: eval_curves_v2.png")

    # Lead time bar chart
    if all_lead_times:
        fig, ax = plt.subplots(figsize=(12, 5))
        ids = [lt["flare_id"] for lt in all_lead_times]
        vals = [lt["lead_time_min"] for lt in all_lead_times]
        colors = ["#4CAF50" if v > 0 else "#F44336" for v in vals]
        bars = ax.bar(range(len(ids)), vals, color=colors)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([f"{lt['flare_id']}\n({lt['class']})" for lt in all_lead_times],
                          fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("Lead Time (minutes)", fontsize=11)
        ax.set_title("Forecast Lead Time per Flare (green=detected, red=missed)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("lead_times_v2.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: lead_times_v2.png")

    # Save results
    results = {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tpr": tpr, "far": far,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "lead_times": all_lead_times,
        "n_flares_detected": len(detected),
        "n_flares_total": len(all_lead_times),
        "threshold": final_thresh,
        "feature_names": feature_names,
    }
    with open("forecast_results_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Saved: forecast_results_v2.json")

    print("\n" + "=" * 60)
    print("[OK] Forecasting v2 complete!")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Flares detected: {len(detected)}/{len(all_lead_times)}")
    if detected:
        print(f"  Avg lead time: {np.mean(lead_vals):.1f} min")
    print(f"  Optimal threshold: {final_thresh:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
