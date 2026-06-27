# Lightweight TinyML-Based Anomaly Detection in Industrial Embedded Systems

**Dissertation Project by Thondupu Dileep**

Simple step-by-step implementation guide.

## Project Overview
Unsupervised anomaly detection using a lightweight Convolutional Autoencoder on industrial vibration data (CWRU + MIMII datasets) with model compression for TinyML deployment.

## Setup
```bash
# 1. Clone or create project
mkdir tinyML-anomaly-detection && cd tinyML-anomaly-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow tensorflow-lite numpy pandas scipy matplotlib scikit-learn jupyter
```

## Step-by-Step Implementation

| Step | Description | Status | Date |
|------|-------------|--------|------|
| 1 | Project Setup & Folder Structure | Pending | - |
| 2 | Download & Explore Datasets (CWRU + MIMII) | Pending | - |
| 3 | Data Preprocessing & Feature Engineering | Pending | - |
| 4 | Build & Train Convolutional Autoencoder | Pending | - |
| 5 | Model Optimization (Pruning + Quantization) | Pending | - |
| 6 | Convert to TFLite & Benchmark | Pending | - |
| 7 | Edge Impulse Simulation (optional) | Pending | - |
| 8 | Results Analysis & Visualization | Pending | - |
| 9 | Write Dissertation Chapters | Pending | - |
| 10 | Final Review & Submission | Pending | - |

**How to track progress:**
- After completing each step, update the `STATUS.md` file.
- Mark status as `Completed` with date.

## Progress Tracking
Run this to check current status:
```bash
cat STATUS.md
```

## Folder Structure
```
.
├── data/                  # Datasets
├── notebooks/             # EDA & experiments
├── src/                   # Main scripts
├── models/                # Saved models
├── results/               # Plots & metrics
├── README.md
├── STATUS.md              # Progress tracker
└── requirements.txt
```

---

## Detailed Steps

### Step 1: Project Setup
- Create all folders
- Generate `requirements.txt`
- Create initial `STATUS.md`

**Command to run after completion:**
```bash
echo "Step 1: Completed - $(date)" >> STATUS.md
```

### Step 2: Datasets
- Download CWRU Bearing Dataset
- Download MIMII Dataset
- Place in `data/raw/`

### Step 3: Preprocessing
- Run notebooks/03_data_preprocessing.ipynb

### Step 4: Model Training
```bash
python src/train.py
```

**Continue similarly for other steps.**

---

**Update STATUS.md after each major step for tracking.**
