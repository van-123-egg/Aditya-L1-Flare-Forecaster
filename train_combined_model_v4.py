"""
Combined Solar Flare Forecasting Model v4 (SoLEXS + HEL1OS)
=============================================================
Self-contained training script for Google Colab.

Trains two models:
  Model A: XGBoost (tabular features) - fast, no GPU needed
  Model B: 1D-CNN + BiLSTM (time series) - benefits from GPU

Usage (on Colab):
  1. Upload training_data_v4.npz to Colab
  2. Upload this script to Colab
  3. Run: !pip install xgboost torch scikit-learn matplotlib
  4. Run: !python train_combined_model_v4.py

Output files (download these back to your PC):
  - forecast_model_v4_xgb.pkl      (XGBoost model)
  - forecast_model_v4_lstm.pt       (PyTorch LSTM model weights)
  - forecast_model_v4_lstm_config.json (LSTM model config)
  - forecast_results_v4.json        (evaluation metrics)
  - eval_curves_v4.png              (ROC/PR curves)
  - feature_importance_v4.png       (XGBoost feature importance)
  - lead_times_v4.png               (per-flare lead times)
  - model_comparison_v4.png         (v3 vs v4 comparison)
"""
import numpy as np
import json
import pickle
import os
import time as timer
import warnings
warnings.filterwarnings("ignore")

# Check if running on Colab
try:
    from google.colab import files as colab_files
    ON_COLAB = True
    print("[*] Running on Google Colab")
except ImportError:
    ON_COLAB = False
    print("[*] Running locally")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, average_precision_score, f1_score
)

import xgboost as xgb

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] PyTorch device: {device}")
except ImportError:
    HAS_TORCH = False
    print("[!] PyTorch not available - skipping LSTM model")


# ==================================================================
# CONFIGURATION
# ==================================================================
DATA_FILE = "training_data_v4.npz"

# XGBoost hyperparameters
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.05,
    reg_lambda=1.0,
    eval_metric="aucpr",
    random_state=42,
    verbosity=0,
)

# LSTM hyperparameters
LSTM_PARAMS = dict(
    n_conv_filters=64,
    n_lstm_hidden=128,
    n_lstm_layers=2,
    dropout=0.3,
    batch_size=256,
    epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
)


# ==================================================================
# DATA LOADING
# ==================================================================
def load_data():
    """Load the prepared .npz training data."""
    print(f"\n[1] Loading {DATA_FILE}...")
    data = np.load(DATA_FILE, allow_pickle=True)

    X_tabular = data["X_tabular"]
    X_ts = data["X_ts"]
    y = data["y"]
    day_idx = data["day_idx"]
    sample_times = data["sample_times"]
    feature_names = list(data["feature_names"])
    day_dates = list(data["day_dates"])
    channel_names = list(data["channel_names"])
    config = json.loads(str(data["config_json"]))
    catalog = json.loads(str(data["catalog_json"]))

    print(f"  Tabular:  {X_tabular.shape} ({len(feature_names)} features)")
    print(f"  TimeSer:  {X_ts.shape} ({len(channel_names)} channels)")
    print(f"  Labels:   {y.sum()} pos / {len(y) - y.sum()} neg "
          f"({100 * y.sum() / len(y):.1f}%)")
    print(f"  Days:     {len(day_dates)} -> {day_dates}")

    return {
        "X_tab": X_tabular, "X_ts": X_ts, "y": y,
        "day_idx": day_idx, "sample_times": sample_times,
        "feature_names": feature_names, "day_dates": day_dates,
        "channel_names": channel_names, "config": config,
        "catalog": catalog,
    }


# ==================================================================
# UTILITY FUNCTIONS
# ==================================================================
def find_optimal_threshold(y_true, y_prob, beta=2.0):
    """Find threshold maximizing F-beta (biased toward recall)."""
    best_score = 0
    best_thresh = 0.3
    for thresh in np.arange(0.05, 0.90, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        fbeta = (1 + beta**2) * prec * rec / max((beta**2 * prec + rec), 1e-8)
        if fbeta > best_score:
            best_score = fbeta
            best_thresh = thresh
    return best_thresh, best_score


def compute_lead_times(y_prob, sample_times, catalog_entries, day_date,
                       threshold, horizon_sec):
    """Compute per-flare lead times."""
    lead_times = []
    for entry in catalog_entries:
        if entry["date"] != day_date:
            continue

        peak_utc = entry.get("slx_peak_utc") or entry.get("hel_peak_utc")
        if peak_utc is None:
            continue

        from datetime import datetime
        try:
            peak_dt = datetime.fromisoformat(peak_utc)
        except (ValueError, TypeError):
            continue

        peak_sec = peak_dt.hour * 3600 + peak_dt.minute * 60 + peak_dt.second
        fid = entry.get("combined_id", "?")
        fclass = entry.get("combined_class", "?")
        source = entry.get("detection_source", "?")

        # Look at 30 min before peak
        pre_mask = (sample_times >= peak_sec - 1800) & (sample_times < peak_sec)
        if pre_mask.sum() == 0:
            continue

        pre_probs = y_prob[pre_mask]
        pre_times = sample_times[pre_mask]
        alert_mask = pre_probs >= threshold

        if alert_mask.sum() > 0:
            first_alert = pre_times[np.where(alert_mask)[0][0]]
            lead_min = (peak_sec - first_alert) / 60.0
            lead_times.append({
                "flare_id": fid, "class": fclass, "source": source,
                "lead_time_min": lead_min, "date": day_date,
            })
        else:
            lead_times.append({
                "flare_id": fid, "class": fclass, "source": source,
                "lead_time_min": 0, "date": day_date,
            })

    return lead_times


# ==================================================================
# MODEL A: XGBoost
# ==================================================================
def train_xgboost(data):
    """Train XGBoost with leave-one-day-out cross-validation."""
    print(f"\n{'=' * 70}")
    print("  MODEL A: XGBoost (Tabular Features)")
    print(f"{'=' * 70}")

    X = data["X_tab"]
    y = data["y"]
    day_idx = data["day_idx"]
    sample_times = data["sample_times"]
    feature_names = data["feature_names"]
    day_dates = data["day_dates"]
    catalog = data["catalog"]
    config = data["config"]
    n_days = len(day_dates)

    all_y_true, all_y_prob, all_y_pred = [], [], []
    all_lead_times = []

    print(f"\n  Leave-One-Day-Out CV ({n_days} folds)")
    print(f"  {'-' * 50}")

    for test_i in range(n_days):
        train_mask = day_idx != test_i
        test_mask = day_idx == test_i

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        times_test = sample_times[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        model = xgb.XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=n_neg / max(n_pos, 1) * 0.7,
        )
        model.fit(X_train_s, y_train, verbose=False)

        y_prob = model.predict_proba(X_test_s)[:, 1]

        # Find optimal threshold on train set
        y_train_prob = model.predict_proba(X_train_s)[:, 1]
        opt_thresh, _ = find_optimal_threshold(y_train, y_train_prob)
        y_pred = (y_prob >= opt_thresh).astype(int)

        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tpr = tp / max(tp + fn, 1)

        print(f"  {day_dates[test_i]}: thresh={opt_thresh:.2f} "
              f"TP={tp} FP={fp} FN={fn} TPR={tpr:.3f}")

        # Lead times
        lt = compute_lead_times(y_prob, times_test, catalog,
                                day_dates[test_i], opt_thresh,
                                config["horizon_sec"])
        for entry in lt:
            status = f"{entry['lead_time_min']:.1f}min" if entry["lead_time_min"] > 0 else "MISSED"
            print(f"    {entry['flare_id']} ({entry['class']}, {entry['source']}): {status}")
        all_lead_times.extend(lt)

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)

    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    roc_auc = roc_auc_score(all_y_true, all_y_prob) if all_y_true.sum() > 0 else 0
    pr_auc = average_precision_score(all_y_true, all_y_prob) if all_y_true.sum() > 0 else 0

    print(f"\n  XGBoost Aggregate Results:")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=["No Flare", "Flare"], zero_division=0))
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")

    detected = [lt for lt in all_lead_times if lt["lead_time_min"] > 0]
    if detected:
        lead_vals = [lt["lead_time_min"] for lt in detected]
        print(f"  Flares detected: {len(detected)}/{len(all_lead_times)}")
        print(f"  Avg lead time: {np.mean(lead_vals):.1f} min")
        print(f"  Median lead:   {np.median(lead_vals):.1f} min")

    # Train final model on all data
    print(f"\n  Training final XGBoost on all data...")
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X)

    final_model = xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1) * 0.7,
    )
    final_model.fit(X_all_s, y, verbose=False)

    y_all_prob = final_model.predict_proba(X_all_s)[:, 1]
    final_thresh, _ = find_optimal_threshold(y, y_all_prob)

    # Save model
    model_path = "forecast_model_v4_xgb.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "scaler": scaler_final,
            "feature_names": feature_names,
            "threshold": final_thresh,
            "horizon_sec": config["horizon_sec"],
            "history_window": config["history_window"],
            "energy_bands": config["energy_bands"],
            "version": "v4_xgb",
        }, f)
    print(f"  Saved: {model_path} (threshold={final_thresh:.2f})")

    # Plots
    _plot_xgb_results(final_model, feature_names, all_y_true, all_y_prob,
                      all_lead_times, roc_auc, pr_auc)

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "lead_times": all_lead_times,
        "y_true": all_y_true, "y_prob": all_y_prob,
        "threshold": final_thresh,
    }


def _plot_xgb_results(model, feature_names, y_true, y_prob,
                       lead_times, roc_auc, pr_auc):
    """Generate XGBoost evaluation plots."""
    # Feature importance
    importance = model.feature_importances_
    top_n = min(30, len(feature_names))
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = []
    for idx in indices:
        name = feature_names[idx]
        if name.startswith("hel_") or name.startswith("cross_"):
            colors.append("#9C27B0")  # purple = HEL1OS
        elif name.startswith("czt_"):
            colors.append("#FF5722")  # deep orange = CZT
        elif "hard" in name.lower():
            colors.append("#F44336")  # red
        elif "soft" in name.lower():
            colors.append("#2196F3")  # blue
        elif "medium" in name.lower():
            colors.append("#FF9800")  # orange
        else:
            colors.append("#607D8B")  # gray
    ax.barh(range(len(indices)), importance[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=7)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top Features v4 (purple=HEL1OS, red=hard, blue=soft, orange=medium)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("feature_importance_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: feature_importance_v4.png")

    # Lead time chart
    if lead_times:
        fig, ax = plt.subplots(figsize=(16, 5))
        ids = [lt["flare_id"] for lt in lead_times]
        vals = [lt["lead_time_min"] for lt in lead_times]
        colors_lt = []
        for lt in lead_times:
            if lt["lead_time_min"] > 0 and lt["source"] == "SoLEXS+HEL1OS":
                colors_lt.append("#9C27B0")  # matched = purple
            elif lt["lead_time_min"] > 0:
                colors_lt.append("#4CAF50")  # detected = green
            else:
                colors_lt.append("#F44336")  # missed = red
        ax.bar(range(len(ids)), vals, color=colors_lt)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([f"{lt['flare_id']}\n({lt['class']})" for lt in lead_times],
                          fontsize=4, rotation=45, ha="right")
        ax.set_ylabel("Lead Time (min before peak)")
        ax.set_title("Forecast Lead Time per Flare (v4 - Combined SoLEXS+HEL1OS)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("lead_times_v4.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: lead_times_v4.png")

    # ROC + PR curves
    if roc_auc > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
        ax1.plot(fpr_arr, tpr_arr, color="#9C27B0", linewidth=2,
                label=f"v4 AUC={roc_auc:.3f}")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.set_title("ROC Curve (v4)", fontweight="bold")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(rec, prec, color="#FF5722", linewidth=2,
                label=f"v4 AP={pr_auc:.3f}")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title("PR Curve (v4)", fontweight="bold")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("eval_curves_v4.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: eval_curves_v4.png")


# ==================================================================
# MODEL B: 1D-CNN + BiLSTM (PyTorch)
# ==================================================================
if HAS_TORCH:
    class FlareDataset(Dataset):
        """PyTorch dataset for time-series flare data."""
        def __init__(self, X_ts, y):
            # X_ts: (N, T, C) -> PyTorch wants (N, C, T) for Conv1d
            self.X = torch.FloatTensor(X_ts).permute(0, 2, 1)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class FlareCNNLSTM(nn.Module):
        """
        Dual-branch 1D-CNN + BiLSTM for solar flare prediction.

        Architecture:
          Branch 1 (SoLEXS):  Conv1D -> Conv1D -> BiLSTM
          Branch 2 (HEL1OS):  Conv1D -> Conv1D -> BiLSTM
          Fusion:             Concat -> Dense -> Dense -> Sigmoid
        """
        def __init__(self, n_slx_channels=5, n_hel_channels=2,
                     n_conv_filters=64, n_lstm_hidden=128,
                     n_lstm_layers=2, dropout=0.3):
            super().__init__()

            # SoLEXS branch (channels 0-4: total, soft, medium, hard, vhard)
            self.slx_conv = nn.Sequential(
                nn.Conv1d(n_slx_channels, n_conv_filters, kernel_size=7, padding=3),
                nn.BatchNorm1d(n_conv_filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(n_conv_filters, n_conv_filters * 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(n_conv_filters * 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            )
            self.slx_lstm = nn.LSTM(
                n_conv_filters * 2, n_lstm_hidden,
                num_layers=n_lstm_layers, batch_first=True,
                bidirectional=True, dropout=dropout if n_lstm_layers > 1 else 0,
            )

            # HEL1OS branch (channels 5-6: CdTe, CZT)
            self.hel_conv = nn.Sequential(
                nn.Conv1d(n_hel_channels, n_conv_filters, kernel_size=7, padding=3),
                nn.BatchNorm1d(n_conv_filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(n_conv_filters, n_conv_filters * 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(n_conv_filters * 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            )
            self.hel_lstm = nn.LSTM(
                n_conv_filters * 2, n_lstm_hidden,
                num_layers=n_lstm_layers, batch_first=True,
                bidirectional=True, dropout=dropout if n_lstm_layers > 1 else 0,
            )

            # Fusion head
            fusion_size = n_lstm_hidden * 2 * 2  # BiLSTM * 2 branches
            self.classifier = nn.Sequential(
                nn.Linear(fusion_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            # x shape: (batch, channels, time_steps)
            # Split into SoLEXS (ch 0-4) and HEL1OS (ch 5-6)
            x_slx = x[:, :5, :]   # (B, 5, T)
            x_hel = x[:, 5:, :]   # (B, 2, T)

            # SoLEXS branch
            slx_conv = self.slx_conv(x_slx)  # (B, 128, T//4)
            slx_seq = slx_conv.permute(0, 2, 1)  # (B, T//4, 128)
            slx_out, _ = self.slx_lstm(slx_seq)  # (B, T//4, 256)
            slx_feat = slx_out[:, -1, :]  # last time step: (B, 256)

            # HEL1OS branch
            hel_conv = self.hel_conv(x_hel)
            hel_seq = hel_conv.permute(0, 2, 1)
            hel_out, _ = self.hel_lstm(hel_seq)
            hel_feat = hel_out[:, -1, :]

            # Fusion
            combined = torch.cat([slx_feat, hel_feat], dim=1)  # (B, 512)
            logit = self.classifier(combined)  # (B, 1)

            return logit.squeeze(1)


def train_lstm(data):
    """Train CNN+BiLSTM with leave-one-day-out CV."""
    if not HAS_TORCH:
        print("\n[!] Skipping LSTM model (PyTorch not available)")
        return None

    print(f"\n{'=' * 70}")
    print("  MODEL B: 1D-CNN + BiLSTM (Time Series)")
    print(f"{'=' * 70}")

    X_ts = data["X_ts"]
    y = data["y"]
    day_idx = data["day_idx"]
    sample_times = data["sample_times"]
    day_dates = data["day_dates"]
    catalog = data["catalog"]
    config = data["config"]
    n_days = len(day_dates)
    p = LSTM_PARAMS

    all_y_true, all_y_prob = [], []
    all_lead_times = []

    print(f"\n  Leave-One-Day-Out CV ({n_days} folds)")
    print(f"  {'-' * 50}")

    for test_i in range(n_days):
        train_mask = day_idx != test_i
        test_mask = day_idx == test_i

        X_train_ts, y_train = X_ts[train_mask], y[train_mask]
        X_test_ts, y_test = X_ts[test_mask], y[test_mask]
        times_test = sample_times[test_mask]

        # Normalize per-channel (fit on train)
        n_channels = X_train_ts.shape[2]
        ch_means = np.zeros(n_channels)
        ch_stds = np.ones(n_channels)
        for c in range(n_channels):
            ch_means[c] = X_train_ts[:, :, c].mean()
            ch_stds[c] = max(X_train_ts[:, :, c].std(), 1e-6)
            X_train_ts[:, :, c] = (X_train_ts[:, :, c] - ch_means[c]) / ch_stds[c]
            X_test_ts[:, :, c] = (X_test_ts[:, :, c] - ch_means[c]) / ch_stds[c]

        # Create datasets
        train_ds = FlareDataset(X_train_ts, y_train)
        test_ds = FlareDataset(X_test_ts, y_test)

        # Class weighting
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1) * 0.7]).to(device)

        train_loader = DataLoader(train_ds, batch_size=p["batch_size"],
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=p["batch_size"], shuffle=False)

        # Build model
        model = FlareCNNLSTM(
            n_slx_channels=5, n_hel_channels=2,
            n_conv_filters=p["n_conv_filters"],
            n_lstm_hidden=p["n_lstm_hidden"],
            n_lstm_layers=p["n_lstm_layers"],
            dropout=p["dropout"],
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(),
                                       lr=p["lr"],
                                       weight_decay=p["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=p["epochs"])

        # Train
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(p["epochs"]):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 7:
                break

        # Load best model and evaluate
        model.load_state_dict(best_state)
        model.eval()

        y_prob_list = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob_list.append(probs)

        y_prob = np.concatenate(y_prob_list)

        # Threshold from train set
        y_train_prob_list = []
        model.eval()
        train_eval_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=False)
        with torch.no_grad():
            for X_batch, _ in train_eval_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_train_prob_list.append(probs)
        y_train_prob = np.concatenate(y_train_prob_list)
        opt_thresh, _ = find_optimal_threshold(y_train, y_train_prob)

        y_pred = (y_prob >= opt_thresh).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tpr = tp / max(tp + fn, 1)

        print(f"  {day_dates[test_i]}: epochs={epoch + 1} thresh={opt_thresh:.2f} "
              f"TP={tp} FP={fp} FN={fn} TPR={tpr:.3f}")

        # Lead times
        lt = compute_lead_times(y_prob, times_test, catalog,
                                day_dates[test_i], opt_thresh,
                                config["horizon_sec"])
        for entry in lt:
            status = f"{entry['lead_time_min']:.1f}min" if entry["lead_time_min"] > 0 else "MISSED"
            print(f"    {entry['flare_id']} ({entry['class']}, {entry['source']}): {status}")
        all_lead_times.extend(lt)

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    # Aggregate
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    roc_auc = roc_auc_score(all_y_true, all_y_prob) if all_y_true.sum() > 0 else 0
    pr_auc = average_precision_score(all_y_true, all_y_prob) if all_y_true.sum() > 0 else 0

    final_thresh, _ = find_optimal_threshold(all_y_true, all_y_prob)
    all_y_pred = (all_y_prob >= final_thresh).astype(int)

    print(f"\n  LSTM Aggregate Results:")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=["No Flare", "Flare"], zero_division=0))
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")

    detected = [lt for lt in all_lead_times if lt["lead_time_min"] > 0]
    if detected:
        lead_vals = [lt["lead_time_min"] for lt in detected]
        print(f"  Flares detected: {len(detected)}/{len(all_lead_times)}")
        print(f"  Avg lead time: {np.mean(lead_vals):.1f} min")

    # Train final model on ALL data
    print(f"\n  Training final LSTM on all data...")

    # Normalize all data
    n_channels = X_ts.shape[2]
    ch_means_final = np.zeros(n_channels)
    ch_stds_final = np.ones(n_channels)
    X_ts_norm = X_ts.copy()
    for c in range(n_channels):
        ch_means_final[c] = X_ts_norm[:, :, c].mean()
        ch_stds_final[c] = max(X_ts_norm[:, :, c].std(), 1e-6)
        X_ts_norm[:, :, c] = (X_ts_norm[:, :, c] - ch_means_final[c]) / ch_stds_final[c]

    all_ds = FlareDataset(X_ts_norm, y)
    all_loader = DataLoader(all_ds, batch_size=p["batch_size"],
                            shuffle=True, drop_last=True)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1) * 0.7]).to(device)

    final_model = FlareCNNLSTM(
        n_slx_channels=5, n_hel_channels=2,
        n_conv_filters=p["n_conv_filters"],
        n_lstm_hidden=p["n_lstm_hidden"],
        n_lstm_layers=p["n_lstm_layers"],
        dropout=p["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(final_model.parameters(),
                                   lr=p["lr"],
                                   weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=p["epochs"])

    for epoch in range(p["epochs"]):
        final_model.train()
        for X_batch, y_batch in all_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = final_model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Save PyTorch model
    torch.save(final_model.state_dict(), "forecast_model_v4_lstm.pt")

    # Save config for inference
    lstm_config = {
        "version": "v4_lstm",
        "n_slx_channels": 5,
        "n_hel_channels": 2,
        "n_conv_filters": p["n_conv_filters"],
        "n_lstm_hidden": p["n_lstm_hidden"],
        "n_lstm_layers": p["n_lstm_layers"],
        "dropout": p["dropout"],
        "threshold": float(final_thresh),
        "ch_means": ch_means_final.tolist(),
        "ch_stds": ch_stds_final.tolist(),
        "channel_names": data["channel_names"],
        "ts_downsample": config["ts_downsample"],
        "history_window": config["history_window"],
        "horizon_sec": config["horizon_sec"],
    }
    with open("forecast_model_v4_lstm_config.json", "w") as f:
        json.dump(lstm_config, f, indent=2)

    print(f"  Saved: forecast_model_v4_lstm.pt")
    print(f"  Saved: forecast_model_v4_lstm_config.json")

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "lead_times": all_lead_times,
        "y_true": all_y_true, "y_prob": all_y_prob,
        "threshold": final_thresh,
    }


# ==================================================================
# COMPARISON PLOTS
# ==================================================================
def plot_comparison(xgb_results, lstm_results):
    """Generate comparison plots between models."""
    print(f"\n  Generating comparison plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC curves
    ax = axes[0]
    for name, res, color in [
        ("XGBoost v4", xgb_results, "#9C27B0"),
        ("CNN+LSTM v4", lstm_results, "#FF5722") if lstm_results else (None, None, None),
    ]:
        if res is None:
            continue
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. PR curves
    ax = axes[1]
    for name, res, color in [
        ("XGBoost v4", xgb_results, "#9C27B0"),
        ("CNN+LSTM v4", lstm_results, "#FF5722") if lstm_results else (None, None, None),
    ]:
        if res is None:
            continue
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name} (AP={res['pr_auc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Lead time comparison
    ax = axes[2]
    models_lt = [("XGBoost", xgb_results)]
    if lstm_results:
        models_lt.append(("CNN+LSTM", lstm_results))

    bar_width = 0.35
    for i, (name, res) in enumerate(models_lt):
        detected = [lt for lt in res["lead_times"] if lt["lead_time_min"] > 0]
        total = len(res["lead_times"])
        if detected:
            vals = [lt["lead_time_min"] for lt in detected]
            metrics = {
                "Det. Rate": len(detected) / max(total, 1),
                "Avg Lead": np.mean(vals),
                "Med Lead": np.median(vals),
            }
        else:
            metrics = {"Det. Rate": 0, "Avg Lead": 0, "Med Lead": 0}

        x = np.arange(len(metrics))
        bars = ax.bar(x + i * bar_width, list(metrics.values()),
                      bar_width, label=name, alpha=0.8)

    ax.set_xticks(np.arange(len(metrics)) + bar_width / 2)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("model_comparison_v4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison_v4.png")


# ==================================================================
# MAIN
# ==================================================================
def main():
    t_start = timer.time()
    print("=" * 70)
    print("  SOLAR FLARE FORECASTING v4 (SoLEXS + HEL1OS Combined)")
    print("=" * 70)

    # Load data
    data = load_data()

    # Train XGBoost
    xgb_results = train_xgboost(data)

    # Train LSTM
    lstm_results = train_lstm(data)

    # Comparison
    plot_comparison(xgb_results, lstm_results)

    # Save combined results
    results = {
        "version": "v4",
        "xgb": {
            "roc_auc": xgb_results["roc_auc"],
            "pr_auc": xgb_results["pr_auc"],
            "threshold": xgb_results["threshold"],
            "lead_times": xgb_results["lead_times"],
            "n_detected": len([lt for lt in xgb_results["lead_times"] if lt["lead_time_min"] > 0]),
            "n_total": len(xgb_results["lead_times"]),
        },
    }
    if lstm_results:
        results["lstm"] = {
            "roc_auc": lstm_results["roc_auc"],
            "pr_auc": lstm_results["pr_auc"],
            "threshold": lstm_results["threshold"],
            "lead_times": lstm_results["lead_times"],
            "n_detected": len([lt for lt in lstm_results["lead_times"] if lt["lead_time_min"] > 0]),
            "n_total": len(lstm_results["lead_times"]),
        }

    with open("forecast_results_v4.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  Saved: forecast_results_v4.json")

    elapsed = timer.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  [OK] Training complete in {elapsed:.0f} seconds!")
    print(f"{'=' * 70}")

    # Summary
    print(f"\n  XGBoost: ROC-AUC={xgb_results['roc_auc']:.4f}, "
          f"PR-AUC={xgb_results['pr_auc']:.4f}")
    if lstm_results:
        print(f"  LSTM:    ROC-AUC={lstm_results['roc_auc']:.4f}, "
              f"PR-AUC={lstm_results['pr_auc']:.4f}")

    # Download files on Colab
    download_files = [
        "forecast_model_v4_xgb.pkl",
        "forecast_results_v4.json",
        "eval_curves_v4.png",
        "feature_importance_v4.png",
        "lead_times_v4.png",
        "model_comparison_v4.png",
    ]
    if lstm_results:
        download_files.extend([
            "forecast_model_v4_lstm.pt",
            "forecast_model_v4_lstm_config.json",
        ])

    print(f"\n  FILES TO DOWNLOAD:")
    for f in download_files:
        size = os.path.getsize(f) / 1024 if os.path.exists(f) else 0
        print(f"    - {f} ({size:.0f} KB)")

    if ON_COLAB:
        print(f"\n  Auto-downloading files from Colab...")
        for f in download_files:
            if os.path.exists(f):
                try:
                    colab_files.download(f)
                    print(f"    Downloaded: {f}")
                except Exception as e:
                    print(f"    [!] Download failed for {f}: {e}")
    else:
        print(f"\n  Not on Colab - files saved in current directory.")
        print(f"  Copy these files to your project folder to use with the dashboard.")


if __name__ == "__main__":
    main()
