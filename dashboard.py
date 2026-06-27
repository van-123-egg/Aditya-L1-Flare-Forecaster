import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import pickle
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import uniform_filter1d
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# App Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Aditya-L1 Solar Monitor", layout="wide", page_icon="☀️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;900&display=swap');
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
/* Stunning vibrant background (space/solar theme) */
.stApp {
    background: 
        radial-gradient(circle at 15% 50%, rgba(245, 158, 11, 0.15), transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(225, 29, 72, 0.15), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(76, 29, 149, 0.2), transparent 50%),
        linear-gradient(135deg, #09090b 0%, #171124 100%);
    color: #f8fafc;
}
/* Enhanced Glassmorphism for Sidebar */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(9, 9, 11, 0.6) 0%, rgba(23, 17, 36, 0.8) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255,255,255,0.05);
}
/* Premium Metric Cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.01) 100%);
    border-top: 1px solid rgba(255,255,255,0.2);
    border-left: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    border-color: rgba(245, 158, 11, 0.5);
    box-shadow: 0 15px 40px rgba(245, 158, 11, 0.2);
}
div[data-testid="metric-container"] label {
    color: #cbd5e1 !important;
    font-weight: 500;
    font-size: 1.05rem;
}
/* Vibrant Gradient Headers */
h1, h2, h3 {
    background: linear-gradient(90deg, #f59e0b, #ef4444, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}
.stAlert {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(0,0,0,0.3) !important;
    backdrop-filter: blur(16px);
}
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model():
    with open("forecast_model_v4_xgb.pkl", "rb") as f:
        data = pickle.load(f)
    return data

model_data = load_model()
xgb_model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]
threshold = model_data["threshold"]
HISTORY_WINDOW = model_data["history_window"]
ENERGY_BANDS = model_data["energy_bands"]

from prepare_training_data import engineer_features_v4

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_day_data(date_folder):
    dataset_dir = "dataset"
    folder_path = os.path.join(dataset_dir, date_folder)
    
    # Find files
    search_roots = [folder_path, os.path.join(folder_path, date_folder)]
    lc_path, pi_path = None, None
    for root in search_roots:
        for det in ["SDD2", "SDD1"]:
            sdd = os.path.join(root, det)
            if not os.path.isdir(sdd): continue
            lcs = glob.glob(os.path.join(sdd, "*.lc.gz")) + glob.glob(os.path.join(sdd, "*.lc"))
            pis = glob.glob(os.path.join(sdd, "*.pi.gz")) + glob.glob(os.path.join(sdd, "*.pi"))
            if lcs:
                lc_path = lcs[0]
                if pis: pi_path = pis[0]
                break
        if lc_path: break
        
    if not lc_path: return None
    
    hdul = fits.open(lc_path)
    times = hdul[1].data["TIME"].copy()
    counts = np.nan_to_num(hdul[1].data["COUNTS"].copy(), nan=0.0)
    
    header = hdul[1].header
    mjd_ref = header["MJDREFI"] + header["MJDREFF"]
    tstart = Time(mjd_ref + header["TSTART"] / 86400.0, format="mjd")
    start_dt = tstart.datetime
    
    # Calculate UTC datetime for all points
    dt_times = pd.date_range(start=start_dt, periods=len(times), freq='S')
    hdul.close()
    
    bands = {}
    if pi_path:
        hdul_pi = fits.open(pi_path)
        spec_counts = hdul_pi[1].data["COUNTS"]
        for bname, (ch_lo, ch_hi) in ENERGY_BANDS.items():
            bands[bname] = np.nansum(spec_counts[:, ch_lo:ch_hi], axis=1).astype(np.float64)
        hdul_pi.close()
        
    return {
        "dt": dt_times,
        "raw_sec": times,
        "counts": counts,
        "bands": bands,
        "has_spectrum": len(bands) > 0
    }

@st.cache_data
def load_hel1os(date_str):
    day_folder = f"hel1os/HLS_{date_str}"
    if not os.path.exists(day_folder):
        return None
    
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from detect_flares_hel1os import load_hel1os_day
        data = load_hel1os_day(day_folder)
    finally:
        sys.stdout = old_stdout
        
    if data is None: return None
    
    tstart = Time(data['mjd_ref'], format="mjd")
    start_dt = tstart.datetime
    dt_times = pd.to_datetime(start_dt) + pd.to_timedelta(data['times'], unit='s')
    
    return {
        "dt": dt_times,
        "raw_sec": data["times"],
        "cdte": data["total_ctr"],
        "czt": data["czt_ctr"]
    }

@st.cache_data
def get_aligned_data(date_str):
    """Loads SoLEXS and aligns HEL1OS to its time grid."""
    
    # 1. Find SoLEXS folder
    dataset_dir = "dataset"
    slx_folder = None
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            if f.startswith("AL1_") and date_str in f:
                slx_folder = f
                break
                
    slx_data = load_day_data(slx_folder) if slx_folder else None
    hel_data = load_hel1os(date_str)
    
    # Base response
    resp = {
        "has_solexs": slx_data is not None,
        "has_hel1os": hel_data is not None,
        "has_spectrum": slx_data["has_spectrum"] if slx_data else False,
        "slx_data": slx_data,
        "hel_data": hel_data
    }
    
    if not resp["has_solexs"] and not resp["has_hel1os"]:
        return None
        
    # Interpolate HEL1OS to SoLEXS timestamps if SoLEXS exists
    if resp["has_solexs"]:
        slx_timestamps = slx_data["dt"].astype(np.int64) / 10**9
        if hel_data is not None:
            hel_timestamps = hel_data["dt"].astype(np.int64) / 10**9
            cdte_aligned = np.interp(slx_timestamps, hel_timestamps, hel_data["cdte"], left=0, right=0)
            czt_aligned = np.interp(slx_timestamps, hel_timestamps, hel_data["czt"], left=0, right=0)
        else:
            cdte_aligned = np.zeros(len(slx_timestamps))
            czt_aligned = np.zeros(len(slx_timestamps))
            
        slx_data["hel_cdte"] = cdte_aligned
        slx_data["hel_czt"] = czt_aligned
        
    return resp

# ---------------------------------------------------------
# Feature Engineering (Single Point)
# ---------------------------------------------------------
def extract_features(data, current_idx):
    """Extract features for the window ending at current_idx using v4 logic."""
    # We need a chunk of history. Let's slice the arrays to speed up engineer_features_v4
    # We need at least HISTORY_WINDOW + 600 seconds of history for the derivatives
    start_idx = max(0, current_idx - HISTORY_WINDOW - 600)
    end_idx = current_idx
    
    t_slice = data["raw_sec"][start_idx:end_idx]
    c_slice = data["counts"][start_idx:end_idx]
    b_slice = {k: v[start_idx:end_idx] for k, v in data["bands"].items()}
    
    cdte_slice = data["hel_cdte"][start_idx:end_idx]
    czt_slice = data["hel_czt"][start_idx:end_idx]
    
    df_feat, names, sample_indices, _ = engineer_features_v4(
        t_slice, c_slice, b_slice, data["has_spectrum"], 
        cdte_slice, czt_slice
    )
    
    if len(df_feat) == 0:
        # Fallback if window is too small (e.g. at the very start of observation)
        return np.zeros((1, len(feature_names)))
        
    # Take the LAST computed window (which corresponds to current_idx if we sliced correctly)
    X = df_feat[-1:]
    
    # Ensure column order matches exactly what the model expects
    import pandas as pd
    df_X = pd.DataFrame(X, columns=names)
    # Fill any missing columns with 0
    for col in feature_names:
        if col not in df_X.columns:
            df_X[col] = 0.0
    
    X_ordered = df_X[feature_names].values
    X_scaled = scaler.transform(X_ordered)
    return X_scaled

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("☀️ Aditya-L1 Space Weather: Real-Time Flare Monitor")
st.markdown("Monitor Soft & Hard X-ray emissions dynamically to predict Solar Flares before they peak using a deep learning-powered pipeline.")

# Gather all available dates from both datasets
available_dates = set()
if os.path.exists("dataset"):
    for f in os.listdir("dataset"):
        if f.startswith("AL1_") and len(f.split("_")) > 3:
            available_dates.add(f.split("_")[3])
if os.path.exists("hel1os"):
    for f in os.listdir("hel1os"):
        if f.startswith("HLS_") and len(f.split("_")) > 1:
            available_dates.add(f.split("_")[1])

sorted_dates = sorted(list(available_dates))
formatted_dates = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in sorted_dates]

# Sidebar
st.sidebar.title("Controls")
st.sidebar.markdown("---")
selected_fmt = st.sidebar.selectbox("📅 Select Observation Day", formatted_dates)
selected_date_str = selected_fmt.replace("-", "")

data = get_aligned_data(selected_date_str)

if not data:
    st.error("❌ No data available for this date.")
    st.stop()

has_solexs = data["has_solexs"]
has_hel1os = data["has_hel1os"]

if not has_solexs:
    st.warning("⚠️ No SoLEXS (Soft X-ray) data found for this date. The AI forecasting model requires SoLEXS features.")
elif not data["has_spectrum"]:
    st.warning("⚠️ No spectrum (.pi) file found for SoLEXS. Forecasting requires energy-band features.")

if not has_hel1os:
    st.info("ℹ️ No HEL1OS (Hard X-ray) data found for this date. The forecasting model will substitute zeros for hard X-ray features.")

# Determine time axis for the slider
if has_solexs:
    total_seconds = len(data["slx_data"]["counts"])
    dt_array = data["slx_data"]["dt"]
else:
    total_seconds = len(data["hel_data"]["dt"])
    dt_array = data["hel_data"]["dt"]

min_idx = HISTORY_WINDOW
max_idx = max(total_seconds - 1, min_idx)

if min_idx >= total_seconds:
    st.error("The observation is too short to run the model.")
    st.stop()

current_idx = st.sidebar.slider(
    "⏱️ Simulate Live Time",
    min_value=min_idx,
    max_value=max_idx,
    value=min_idx,
    step=60,
    format="Sec %d"
)

current_dt = dt_array[current_idx]
st.sidebar.markdown(f"**Current UTC Time:**<br><span style='color:#38bdf8; font-size:1.1em;'>{current_dt.strftime('%Y-%m-%d %H:%M:%S')}</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Run inference if SoLEXS is available
prob = 0.0
if has_solexs and data["has_spectrum"]:
    X_scaled = extract_features(data["slx_data"], current_idx)
    prob = xgb_model.predict_proba(X_scaled)[0, 1]

# ---------------------------------------------------------
# Dashboard Body
# ---------------------------------------------------------

# Flare Alert Banner
if has_solexs and data["has_spectrum"]:
    if prob >= threshold:
        slx = data["slx_data"]
        past_counts = slx["counts"][max(0, current_idx - HISTORY_WINDOW):current_idx]
        bg = np.median(past_counts) if len(past_counts) > 0 else 10
        current_counts = past_counts[-1] if len(past_counts) > 0 else 0
        
        # If current flux is massively elevated, the flare is already happening
        if current_counts > max(bg * 4, 150):
            st.markdown(f"<div style='background: rgba(249, 115, 22, 0.2); border-left: 5px solid #f97316; padding: 15px; border-radius: 8px;'><h3 style='margin:0; color:#fdba74;'>🔥 ONGOING FLARE!</h3><p style='margin:0; font-size:1.1em;'>A solar flare is currently in progress. | Current Flux: <b>{current_counts:.0f} cps</b></p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background: rgba(239, 68, 68, 0.2); border-left: 5px solid #ef4444; padding: 15px; border-radius: 8px;'><h3 style='margin:0; color:#fca5a5;'>🚨 FLARE WARNING!</h3><p style='margin:0; font-size:1.1em;'>Model Probability: <b>{prob*100:.1f}%</b> | High probability of a flare peaking in the next 15 minutes!</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background: rgba(16, 185, 129, 0.1); border-left: 5px solid #10b981; padding: 15px; border-radius: 8px;'><h3 style='margin:0; color:#6ee7b7;'>✅ Space Weather is Quiet</h3><p style='margin:0; font-size:1.1em;'>Model Probability: <b>{prob*100:.1f}%</b> (Threshold: {threshold*100:.1f}%)</p></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Layout columns for metrics
if has_solexs and data["has_spectrum"]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Soft X-Rays", f"{data['slx_data']['bands']['soft'][current_idx]:.0f} cps")
    col2.metric("Medium X-Rays", f"{data['slx_data']['bands']['medium'][current_idx]:.0f} cps")
    col3.metric("Hard X-Rays", f"{data['slx_data']['bands']['hard'][current_idx]:.0f} cps")
    col4.metric("Very Hard X-Rays", f"{data['slx_data']['bands']['vhard'][current_idx]:.0f} cps")

# Plotly Charts
st.subheader("Live X-Ray Lightcurves (Last 60 Minutes)")

# Only plot the last 60 minutes up to current_idx
plot_start = current_idx - HISTORY_WINDOW

col_solexs, col_hel1os = st.columns(2)

with col_solexs:
    st.markdown("### 🔵 SoLEXS (Soft X-rays)")
    if has_solexs:
        slx = data["slx_data"]
        plot_times = slx["dt"][plot_start:current_idx]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Total Counts (All Energies)", "Energy Bands (Log Scale)"))

        fig.add_trace(go.Scatter(x=plot_times, y=slx["counts"][plot_start:current_idx], 
                                 mode='lines', name='Total', line=dict(color='white', width=1.5), fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
        
        if data["has_spectrum"]:
            fig.add_trace(go.Scatter(x=plot_times, y=slx["bands"]["soft"][plot_start:current_idx], 
                                     mode='lines', name='Soft', line=dict(color='#38bdf8')), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_times, y=slx["bands"]["medium"][plot_start:current_idx], 
                                     mode='lines', name='Medium', line=dict(color='#fbbf24')), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_times, y=slx["bands"]["hard"][plot_start:current_idx], 
                                     mode='lines', name='Hard', line=dict(color='#ef4444')), row=2, col=1)

        fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                          font=dict(color="#94a3b8"))
        fig.update_yaxes(title_text="Counts/sec", row=1, col=1, gridcolor="rgba(255,255,255,0.1)")
        fig.update_yaxes(title_text="Counts/sec (Log)", type="log", row=2, col=1, gridcolor="rgba(255,255,255,0.1)")
        fig.update_xaxes(title_text="Time (UTC)", row=2, col=1, gridcolor="rgba(255,255,255,0.1)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No SoLEXS data available for this date.")

with col_hel1os:
    st.markdown("### 🟣 HEL1OS (Hard X-rays)")
    
    if has_hel1os:
        if has_solexs:
            # Plotted from interpolated SoLEXS grid
            h_times = data["slx_data"]["dt"][plot_start:current_idx]
            h_cdte = data["slx_data"]["hel_cdte"][plot_start:current_idx]
            h_czt = data["slx_data"]["hel_czt"][plot_start:current_idx]
        else:
            # Plotted from raw HEL1OS grid
            hel = data["hel_data"]
            h_times = hel["dt"][plot_start:current_idx]
            h_cdte = hel["cdte"][plot_start:current_idx]
            h_czt = hel["czt"][plot_start:current_idx]
            
        fig_h = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                              subplot_titles=("CdTe Total (1.8-90 keV)", "CZT Total (18-160 keV)"))
                              
        fig_h.add_trace(go.Scatter(x=h_times, y=h_cdte, mode='lines', name='CdTe', line=dict(color='#a855f7', width=1.5), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'), row=1, col=1)
        
        # Only plot CZT if it has data (not strictly zeros)
        if np.any(h_czt > 0):
            fig_h.add_trace(go.Scatter(x=h_times, y=h_czt, mode='lines', name='CZT', line=dict(color='#f97316', width=1.5), fill='tozeroy', fillcolor='rgba(249, 115, 22, 0.1)'), row=2, col=1)
            
        fig_h.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94a3b8"))
        fig_h.update_yaxes(title_text="Counts/sec", row=1, col=1, gridcolor="rgba(255,255,255,0.1)")
        fig_h.update_yaxes(title_text="Counts/sec", row=2, col=1, gridcolor="rgba(255,255,255,0.1)")
        fig_h.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.warning("No HEL1OS data available for this date.")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b; font-size:0.9em;'>Developed using Aditya-L1 SoLEXS & HEL1OS PRADAN Data for the Bharatiya Antariksh Hackathon 2026</p>", unsafe_allow_html=True)
