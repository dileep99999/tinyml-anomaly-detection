import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")


# settings
NORMAL_DIR   = r"../MIMII/normal"
ABNORMAL_DIR = r"../MIMII/abnormal"
OUTPUT_DIR   = r"../data/processed"
SAMPLE_RATE  = 16000
DURATION     = 10.0
N_MELS       = 64
HOP_LENGTH   = 512
N_FFT        = 1024
TRAIN_RATIO  = 0.8


def to_log_mel(filepath):
    # load audio and convert to log mel spectrogram
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    target = int(SAMPLE_RATE * DURATION)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH,
                                          n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def load_folder(folder, label):
    specs  = []
    labels = []
    files  = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
    print(f"  {folder}: {len(files)} files")
    for fname in files:
        specs.append(to_log_mel(os.path.join(folder, fname)))
        labels.append(label)
    return np.array(specs, dtype=np.float32), np.array(labels, dtype=np.int32)


def process_mimii():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Processing MIMII files...")

    X_normal,   y_normal   = load_folder(NORMAL_DIR,   label=0)
    X_abnormal, y_abnormal = load_folder(ABNORMAL_DIR, label=1)

    # shuffle and split normal data 80/20
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_normal))
    X_normal = X_normal[idx]

    split     = int(len(X_normal) * TRAIN_RATIO)
    X_train   = X_normal[:split]
    X_val     = X_normal[split:]

    # normalise - fit only on training normal data
    mu    = X_train.mean()
    sigma = X_train.std() + 1e-8

    X_train    = ((X_train    - mu) / sigma).astype(np.float32)
    X_val      = ((X_val      - mu) / sigma).astype(np.float32)
    X_abnormal = ((X_abnormal - mu) / sigma).astype(np.float32)

    out = os.path.join(OUTPUT_DIR, "mimii_processed.npz")
    np.savez_compressed(out,
                        X_train=X_train,
                        X_val=X_val,
                        X_abnormal=X_abnormal,
                        norm_mean=np.array([mu]),
                        norm_std=np.array([sigma]))

    print(f"\nSaved to {out}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Abnormal: {len(X_abnormal)}")


if __name__ == "__main__":
    process_mimii()
