import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


PLOTS_DIR    = r"../results/plots"
HISTORY_JSON = r"../results/training_history.json"
METRICS_JSON = r"../results/metrics.json"
BENCH_JSON   = r"../results/benchmark.json"
DATA_PATH    = r"../data/processed/cwru_processed.npz"
KERAS_MODEL  = r"../models/cae_model.h5"


def plot_training():
    with open(HISTORY_JSON) as f:
        h = json.load(f)
    epochs = range(1, len(h["loss"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, h["loss"],     label="Train Loss")
    plt.plot(epochs, h["val_loss"], label="Val Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_loss.png"), dpi=150)
    plt.close()
    print("Saved: training_loss.png")


def plot_model_sizes():
    with open(BENCH_JSON) as f:
        b = json.load(f)

    labels = ["Keras\n(float32)", "TFLite\n(float32)", "TFLite\n(int8)"]
    sizes  = [b["keras_kb"], b["f32_kb"], b["int8_kb"]]

    if b.get("pruned_kb", 0) > 0:
        labels.insert(1, "Pruned\n(float32)")
        sizes.insert(1, b["pruned_kb"])

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, sizes, color=["#2196F3","#673AB7","#009688","#FF5722"][:len(labels)])
    for bar, sz in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{sz:.0f} KB", ha="center", fontsize=9, fontweight="bold")
    plt.axhline(100, color="green", linestyle="--", alpha=0.7, label="Target <100 KB")
    plt.ylabel("Size (KB)")
    plt.title("Model Size Comparison")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_sizes.png"), dpi=150)
    plt.close()
    print("Saved: model_sizes.png")


def plot_latency():
    with open(BENCH_JSON) as f:
        b = json.load(f)

    plt.figure(figsize=(5, 4))
    vals   = [b["f32_ms"], b["int8_ms"]]
    labels = ["TFLite float32", "TFLite int8"]
    bars   = plt.bar(labels, vals, color=["#009688","#FF5722"], width=0.4)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{v:.2f} ms", ha="center", fontsize=9, fontweight="bold")
    plt.axhline(50, color="green", linestyle="--", alpha=0.7, label="Target <50 ms")
    plt.ylabel("Inference Time (ms)")
    plt.title("Inference Latency")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency.png"), dpi=150)
    plt.close()
    print("Saved: latency.png")


def plot_per_fault_mse():
    data   = np.load(DATA_PATH, allow_pickle=True)
    X      = data["X_raw"][..., np.newaxis]
    y_name = data["y_names"]
    model  = tf.keras.models.load_model(KERAS_MODEL)

    recon  = model.predict(X, batch_size=64, verbose=0)
    mse    = np.mean((X - recon) ** 2, axis=(1, 2))

    unique = list(dict.fromkeys(y_name))   # preserve order
    groups = [mse[y_name == n] for n in unique]

    plt.figure(figsize=(11, 4))
    bp = plt.boxplot(groups, patch_artist=True, widths=0.5)
    colors = ["#4CAF50" if n == "Normal" else "#F44336" for n in unique]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    plt.xticks(range(1, len(unique)+1), unique, rotation=30, ha="right", fontsize=8)
    plt.ylabel("Reconstruction MSE")
    plt.title("MSE by Fault Type (CWRU)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_fault_mse.png"), dpi=150)
    plt.close()
    print("Saved: per_fault_mse.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if os.path.exists(HISTORY_JSON):
        plot_training()

    if os.path.exists(BENCH_JSON):
        plot_model_sizes()
        plot_latency()

    if os.path.exists(KERAS_MODEL) and os.path.exists(DATA_PATH):
        plot_per_fault_mse()

    if os.path.exists(METRICS_JSON) and os.path.exists(BENCH_JSON):
        with open(METRICS_JSON) as f: m = json.load(f)
        with open(BENCH_JSON)   as f: b = json.load(f)
        print("\n--- Summary ---")
        print(f"F1: {m['f1']:.4f}  AUC: {m['auc_roc']:.4f}")
        print(f"TFLite int8: {b['int8_kb']:.0f} KB  |  {b['int8_ms']:.2f} ms")
        print(f"Compression: {b['compression']:.1f}x vs Keras H5")

    print("\nDone.")


if __name__ == "__main__":
    main()
