<div align="center">
  <h1>Lightweight TinyML-Based Anomaly Detection<br>for Industrial Embedded Systems</h1>
  <p><strong>A Convolutional Autoencoder approach for real-time predictive maintenance on ultra-constrained Edge MCUs (like Arduino Uno).</strong></p>
</div>

<hr>

## Project Overview

Industrial machinery failures cost billions annually. Traditional cloud-based monitoring suffers from latency, bandwidth costs, and security risks. This project implements a **TinyML anomaly detection system** that runs directly on the edge, enabling real-time predictive maintenance on ultra-low-power microcontrollers.

By utilizing **Unsupervised Convolutional Autoencoders (CAE)**, the system learns the normal operating vibrations and acoustic signatures of machinery. When a machine degrades, the reconstruction error spikes, triggering an anomaly alert.

### Key Features
- **Unsupervised Learning:** Trains only on "Normal" data. No need for hard-to-get fault data.
- **Extreme Quantization (INT8):** Neural networks compressed by 11x (from float32) without losing accuracy.
- **Ultra-low Footprint:** Models optimized to fit within **32 KB Flash & 2 KB RAM** (Arduino Uno/Nano).
- **Multi-Modal:** Supports 1D Vibration Data (CWRU) and 2D Acoustic Spectrograms (MIMII).
- **Real-time Web Dashboard:** A live Flask GUI simulating the sliding-window industrial deployment.

---

## System Architecture

The pipeline consists of three main components:

1. **Data Processing & Feature Extraction**
   - **Vibration (CWRU):** 1D Time-series sliding windows (2048 steps).
   - **Acoustic (MIMII):** 2D Log-Mel Spectrograms (64 Mel-bands × 313 frames).
2. **TinyML Model Training Pipeline**
   - 1D/2D Convolutional Autoencoders and an ultra-lightweight MLP for the Arduino Uno.
   - Post-Training INT8 Quantization via TensorFlow Lite.
3. **Deployment**
   - **Flask Web GUI:** Live simulation of the detection pipeline.
   - **C++ Header Generation:** Directly exports models to `.h` files for bare-metal MCU compilation.

---

## Performance & Metrics

Our INT8 quantized models achieve near-perfect AUC scores while maintaining microsecond inference latencies on Cortex-M processors.

| Model / Dataset | Accuracy (AUC) | Flash Size | RAM Usage | Inference Time (Cortex-M4) |
| :--- | :---: | :---: | :---: | :---: |
| **1D CAE (CWRU)** | 1.000 | 18.2 KB | 5.4 KB | 0.8 ms |
| **2D CAE (MIMII)** | 0.982 | 45.1 KB | 12.8 KB | 4.2 ms |
| **Uno MLP (CWRU)** | 0.985 | **4.2 KB** | **0.8 KB** | **< 1.0 ms** |

> **Note:** The *Uno MLP* uses statistical feature extraction (RMS, Kurtosis, Skewness, Crest Factor) to compress the inputs before passing them into a 186-parameter dense network, allowing it to run on the 8-bit ATmega328P.

---

## Repository Structure

```text
tinyML-anomaly-detection/
├── app/                        # Flask Web GUI (app.py, templates, static)
├── models/                     # Saved Keras (.h5), TFLite, and C++ header models
├── notebooks/                  # EDA, Preprocessing, and Exploration Jupyter Notebooks
├── results/                    # Output plots, confusion matrices, and benchmark JSONs
├── src/                        # Core Python source code
│   ├── preprocess_cwru.py      # CWRU dataset pipeline
│   ├── preprocess_mimii.py     # MIMII spectrogram pipeline
│   ├── model.py                # Keras CAE architectures
│   ├── train.py                # Main training loop for CAEs
│   ├── train_uno_model.py      # Statistical feature MLP for Arduino Uno
│   ├── evaluate.py             # ROC, AUC, and threshold calculation
│   ├── convert_tflite.py       # INT8 Quantization & C++ Header export
│   └── visualise_results.py    # Plot generation
└── requirements.txt            # Python dependencies
```

---

## Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/dileep99999/tinyml-anomaly-detection.git
cd tinyml-anomaly-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Running the Web GUI
To launch the real-time simulation dashboard:
```bash
python app/app.py
```
Then, open your browser and navigate to **`http://localhost:5000`**.

### 3. Training a Model from Scratch
If you want to retrain the Convolutional Autoencoder:
```bash
python src/preprocess_cwru.py    # Process raw data
python src/train.py              # Train the Keras model
python src/evaluate.py           # Evaluate on test set
python src/convert_tflite.py     # Quantize to INT8
```

### 4. Arduino Uno Compilation
To generate the ultra-lightweight statistical model for the Arduino Uno:
```bash
python src/train_uno_model.py
```
This generates `models/uno_anomaly_model.h` which can be dropped directly into the Arduino IDE.

---

## License
This project is open-source and licensed under the **MIT License**.

## Author
**Thondupu Dileep** (BITS Pilani WILP Dissertation)
