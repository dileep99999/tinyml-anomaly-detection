import os
import numpy as np
import scipy.io as sio
from scipy.fft import rfft
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# file paths and settings
CWRU_RAW_DIR = r"../CWRU/raw"
OUTPUT_DIR   = r"../data/processed"
WINDOW_SIZE  = 2048
OVERLAP      = 0.5
SAMPLE_RATE  = 48000

# each mat file mapped to a label number
FILE_LABEL_MAP = {
    "Time_Normal_1_098.mat": (0, "Normal"),
    "B007_1_123.mat"       : (1, "Ball_007"),
    "B014_1_190.mat"       : (2, "Ball_014"),
    "B021_1_227.mat"       : (3, "Ball_021"),
    "IR007_1_110.mat"      : (4, "InnerRace_007"),
    "IR014_1_175.mat"      : (5, "InnerRace_014"),
    "IR021_1_214.mat"      : (6, "InnerRace_021"),
    "OR007_6_1_136.mat"    : (7, "OuterRace_007"),
    "OR014_6_1_202.mat"    : (8, "OuterRace_014"),
    "OR021_6_1_239.mat"    : (9, "OuterRace_021"),
}


def load_signal(filepath):
    # load the mat file and extract drive-end vibration signal
    mat = sio.loadmat(filepath)
    for key in mat.keys():
        if "DE_time" in key:
            return mat[key].flatten().astype(np.float32)
    for key in mat.keys():
        if "time" in key.lower() and not key.startswith("__"):
            return mat[key].flatten().astype(np.float32)
    raise ValueError(f"No signal found in {filepath}")


def make_windows(signal, window_size, overlap):
    # cut signal into overlapping chunks
    step = int(window_size * (1 - overlap))
    windows = []
    i = 0
    while i + window_size <= len(signal):
        windows.append(signal[i:i + window_size])
        i += step
    return np.array(windows, dtype=np.float32)


def get_features(window):
    # time domain stats
    rms   = np.sqrt(np.mean(window ** 2))
    peak  = np.max(np.abs(window))
    stats = np.array([
        np.mean(window),
        np.std(window),
        rms,
        peak,
        peak / (rms + 1e-9),   # crest factor
        kurtosis(window),
        skew(window)
    ], dtype=np.float32)

    # fft magnitude (first 256 bins)
    fft_mag = (np.abs(rfft(window))[:256] / len(window)).astype(np.float32)

    return np.concatenate([stats, fft_mag])


def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_windows = []
    features    = []
    labels      = []
    names       = []

    print("Processing CWRU files...")

    for fname, (label, fault_name) in FILE_LABEL_MAP.items():
        fpath = os.path.join(CWRU_RAW_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  Skipping {fname} - not found")
            continue

        signal  = load_signal(fpath)
        windows = make_windows(signal, WINDOW_SIZE, OVERLAP)
        feat    = np.array([get_features(w) for w in windows], dtype=np.float32)

        print(f"  {fault_name}: {len(windows)} windows")

        raw_windows.append(windows)
        features.append(feat)
        labels.extend([label] * len(windows))
        names.extend([fault_name] * len(windows))

    X_raw  = np.vstack(raw_windows)
    X_feat = np.vstack(features)
    y      = np.array(labels, dtype=np.int32)
    y_name = np.array(names)

    # normalise using only normal data (unsupervised setup)
    mask   = (y == 0)
    scaler = StandardScaler()
    scaler.fit(X_raw[mask])
    X_raw_norm  = scaler.transform(X_raw)

    scaler2 = StandardScaler()
    scaler2.fit(X_feat[mask])
    X_feat_norm = scaler2.transform(X_feat)

    # save scaler params for inference
    np.save(os.path.join(OUTPUT_DIR, "cwru_scaler_mean.npy"),  scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, "cwru_scaler_scale.npy"), scaler.scale_)

    out = os.path.join(OUTPUT_DIR, "cwru_processed.npz")
    np.savez_compressed(out,
                        X_raw=X_raw_norm,
                        X_features=X_feat_norm,
                        y_labels=y,
                        y_names=y_name)

    print(f"\nSaved to {out}")
    print(f"  Shape: {X_raw_norm.shape}")
    print(f"  Normal: {mask.sum()}  |  Fault: {(~mask).sum()}")


if __name__ == "__main__":
    process_all()
