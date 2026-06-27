# Aditya-L1 Space Weather: Architectural Pipeline & Project Defense ☀️

This document serves as the comprehensive architectural breakdown and defense guide for the **Bharatiya Antariksh Hackathon 2026** Solar Flare Forecasting Project. It details the data structures, algorithms, design choices, and anticipated questions for the hackathon judging panel.

---

## 1. Data Architecture & Utilization

### The Structure of the Data
Our project leverages Level-1 (L1) telemetry from two distinct Aditya-L1 instruments. Both datasets are provided in standard astronomical FITS (Flexible Image Transport System) formats.

**SoLEXS (Solar Low Energy X-ray Spectrometer):**
*   **Format**: `.lc` (Lightcurve - Total counts/sec) and `.pi` (Pulse Invariant - Spectral/Energy bands).
*   **Data Content**: Measures soft X-rays. The `.pi` files contain a 2D matrix (Time $\times$ Energy Channels). We chunked the 340 channels into 5 distinct energy bands: `soft`, `medium`, `hard`, `vhard`, and `uvhard`.
*   **Utilization**: Used as the primary time-grid and baseline indicator for solar plasma heating.

**HEL1OS (High Energy L1 Orbiting X-ray Spectrometer):**
*   **Format**: `.fits` lightcurves.
*   **Data Content**: Measures hard X-rays across two detectors: CdTe (Cadmium Telluride, 1.8–90 keV) and CZT (Cadmium Zinc Telluride, 18–160 keV).
*   **Utilization**: Used as the early-warning precursor. Hard X-rays indicate the initial electron acceleration phase of a flare before the plasma heats up.

### How We Merged and Utilized the Data
Because the two instruments operate independently, their timestamps do not naturally align. 
1. We converted the `TSTART` (Modified Julian Date) from both instrument headers into absolute UTC timestamps.
2. We used **Linear Interpolation** (`np.interp`) to map the HEL1OS hard X-ray counts onto the exact second-by-second time grid of the SoLEXS instrument.
3. If HEL1OS data was completely missing for a specific day, the pipeline gracefully substituted an array of zeros, allowing the models to continue functioning solely on Soft X-rays.

### Metrics & Graphs
*   **The Lightcurve Graphs**: The dashboard displays dynamic time-series plots (Lightcurves) mapping Counts/sec over Time. 
    *   *What they provide*: A visual representation of solar intensity. A flare appears as a sharp, sudden spike (impulsive phase) followed by a slow decay. 
    *   *How they are helpful*: By plotting SoLEXS and HEL1OS side-by-side, users can visually verify the **Neupert Effect**—where the hard X-ray (HEL1OS) spikes slightly before or during the steepest rise of the soft X-ray (SoLEXS) curve.
*   **Metrics**: Our primary evaluation metric for the Machine Learning models is **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve) because the dataset is extremely imbalanced (solar flares only occur ~3.8% of the time). ROC-AUC ensures our model is penalized heavily for false positives while maximizing true flare detections.

---

## 2. Design Choices

*   **Choice 1: Using the Neupert Effect as the Core Feature**
    *   *Why*: Machine learning models are only as good as their features. Instead of feeding raw data blindly, we encoded the physics of the Neupert Effect. We calculated the *derivative* (rate of change) of the SoLEXS soft X-rays and created ratios comparing it directly to the absolute intensity of HEL1OS hard X-rays. This physics-informed feature engineering drastically improved model performance.
*   **Choice 2: XGBoost vs Deep Learning**
    *   *Why*: We implemented *both*. 
        *   **XGBoost** was chosen for the live dashboard because it is incredibly fast, lightweight, runs easily on CPUs, and provides feature importances (explainability). 
        *   **CNN-BiLSTM** was chosen as our peak-performance architecture to prove that raw sequence modeling (allowing the neural network to find its own temporal patterns rather than relying on our handcrafted derivatives) yields the highest accuracy (0.785 ROC-AUC).
*   **Choice 3: Glassmorphism Dashboard UI**
    *   *Why*: Space weather monitoring tools are traditionally visually dense and unintuitive. We utilized a modern Glassmorphism design (frosted glass, gradients, dynamic alert banners) built in Streamlit. We deliberately implemented a "Nowcast vs Forecast" banner: if the flare is already heavily spiking, it alerts "Ongoing Flare" rather than redundantly predicting one.

---

## 3. Algorithms in Detail

### A. Flare Detection & Cataloging Algorithm
Before training, we had to build an automated labeler to find historical flares.
1.  **Dynamic Background Subtraction**: Solar background radiation changes daily. We calculate a rolling 30-minute median to establish the baseline background flux.
2.  **Peak Finding**: We apply `scipy.signal.find_peaks` on smoothed data, enforcing minimum prominence thresholds (e.g., peak must be 2x the background).
3.  **Cross-Matching**: The algorithm compares the SoLEXS catalog to the HEL1OS catalog. If a HEL1OS flare peaks within $\pm 10$ minutes of a SoLEXS flare, they are merged into a single event in `master_combined_catalog.json`.

### B. Feature Engineering Algorithm (109 Features)
At every single minute ($T$), the algorithm looks back at the previous 60 minutes of history and calculates:
1.  **Statistical Moments**: Mean, variance, skewness, and kurtosis of the last 60 mins.
2.  **Derivatives**: Using `np.gradient` on heavily smoothed versions of the data to find the current slope of the X-ray flux. A highly positive slope indicates the start of an impulsive phase.
3.  **Cross-Instrument Ratios**: Dividing the current HEL1OS CdTe counts by the SoLEXS Soft band counts.

### C. The Machine Learning Algorithms
**Model 1: XGBoost Classifier**
An ensemble of decision trees. It iteratively builds trees that correct the errors (residuals) of the previous trees. To handle the fact that flares are rare, we utilized the `scale_pos_weight` hyperparameter, which forces the algorithm to penalize the misclassification of a flare 25x more than a false alarm.

**Model 2: 1D-CNN + BiLSTM (Dual-Branch)**
1.  **Input**: A 60-minute sequence of data.
2.  **Dual-Branch CNN**: Branch A applies 1D Convolutions over the 5 SoLEXS channels. Branch B applies 1D Convolutions over the 2 HEL1OS channels. Convolutions act as automated feature extractors, finding local spikes.
3.  **Concatenation**: The outputs of both CNNs are merged.
4.  **BiLSTM (Bidirectional Long Short-Term Memory)**: The merged sequence is fed into a recurrent neural network that reads the timeline both forwards and backwards. This allows the model to understand long-term dependencies (e.g., a slow decay 40 minutes ago affecting the probability of a flare now).
5.  **Output**: A fully connected layer outputs a Sigmoid probability $[0, 1]$.

---

## 4. Possible Questions & Answers for the Judges

**Q1: Why did you use both SoLEXS and HEL1OS instead of just one?**
*Answer:* Soft X-rays (SoLEXS) show the actual heating of the solar atmosphere, which is the main event of the flare. However, Hard X-rays (HEL1OS) represent the initial particle acceleration. Because of the Neupert Effect, Hard X-rays often spike seconds or minutes *before* the Soft X-rays reach their peak. Combining them gives our model a crucial early-warning leading indicator.

**Q2: The dashboard says "Simulate Live Time". How would this work in real production?**
*Answer:* In production, the slider would be replaced by an API endpoint pinging ISRO's telemetry servers every minute. The script would append the newest data point to a running 60-minute array in memory, run our `extract_features` algorithm in milliseconds, and instantly update the probability banner.

**Q3: Your dataset only had ~4% positive flare samples. How did you prevent the model from just guessing "No Flare" every time?**
*Answer:* This is a classic class-imbalance problem. If the model guessed "No Flare", it would be 96% accurate but entirely useless. We fixed this by switching our evaluation metric from Accuracy to ROC-AUC, and by applying Class Weights (`scale_pos_weight` in XGBoost, and `pos_weight` in the PyTorch BCEWithLogitsLoss). This mathematically forces the loss function to heavily penalize missing a flare.

**Q4: If the model predicts a flare, what exactly is it predicting?**
*Answer:* The model is predicting a binary classification: $1$ if the peak of a solar flare will occur exactly within the *next* 15-minute window relative to the current timestamp, and $0$ otherwise. 

**Q5: Why does the dashboard switch from "Flare Warning" to "Ongoing Flare"?**
*Answer:* Predicting a flare when the flux is already 1,000x above background is trivial and unhelpful for a live monitor (that's "Nowcasting"). We implemented logic that checks the current flux against a dynamic 1-hour rolling background. If the flux is currently $>4x$ the background, the UI shifts context to alert the user that the event is already in progress, making the UX much more intuitive.
