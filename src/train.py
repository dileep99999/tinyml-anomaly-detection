import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
from model import build_cwru_autoencoder


# config
DATA_PATH  = r"../data/processed/cwru_processed.npz"
MODEL_DIR  = r"../models"
RESULT_DIR = r"../results"
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 1e-3
BOTTLENECK = 32


def load_normal(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X    = data["X_raw"]
    y    = data["y_labels"]

    X_normal = X[y == 0][..., np.newaxis]   # only normal for training
    X_fault  = X[y != 0][..., np.newaxis]
    y_fault  = y[y != 0]

    print(f"Normal: {len(X_normal)}  |  Fault: {len(X_fault)}")
    return X_normal, X_fault, y_fault


def train():
    tf.random.set_seed(42)
    np.random.seed(42)

    print("Loading data...")
    X_normal, X_fault, _ = load_normal(DATA_PATH)

    # 80/20 split
    n_train = int(len(X_normal) * 0.8)
    idx     = np.random.permutation(len(X_normal))
    X_train = X_normal[idx[:n_train]]
    X_val   = X_normal[idx[n_train:]]
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # build model
    model = build_cwru_autoencoder(input_length=2048, bottleneck=BOTTLENECK)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse", metrics=["mae"])
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    ckpt_path = os.path.join(MODEL_DIR, "cae_model.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
    ]

    print("Training...")
    t0      = time.time()
    history = model.fit(X_train, X_train,
                        validation_data=(X_val, X_val),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        verbose=1)
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    # save history
    with open(os.path.join(RESULT_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # plot loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"],     label="Train")
    plt.plot(history.history["val_loss"], label="Val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.join(RESULT_DIR, "plots"), exist_ok=True)
    plt.savefig(os.path.join(RESULT_DIR, "plots", "training_loss.png"), dpi=150)
    plt.close()

    # quick check: normal vs fault MSE
    model.load_weights(ckpt_path)
    r_normal = model.predict(X_normal, batch_size=BATCH_SIZE, verbose=0)
    r_fault  = model.predict(X_fault,  batch_size=BATCH_SIZE, verbose=0)
    mse_n    = np.mean((X_normal - r_normal) ** 2, axis=(1, 2))
    mse_f    = np.mean((X_fault  - r_fault)  ** 2, axis=(1, 2))
    print(f"\nNormal MSE mean: {mse_n.mean():.6f}")
    print(f"Fault  MSE mean: {mse_f.mean():.6f}")
    print(f"Ratio: {mse_f.mean()/mse_n.mean():.1f}x")


if __name__ == "__main__":
    train()
