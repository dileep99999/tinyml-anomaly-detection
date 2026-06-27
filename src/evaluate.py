import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf


# paths
DATA_PATH    = r"../data/processed/cwru_processed.npz"
KERAS_MODEL  = r"../models/cae_model.h5"
TFLITE_MODEL = r"../models/cae_quantized.tflite"
RESULTS_DIR  = r"../results"
PLOTS_DIR    = r"../results/plots"
BATCH_SIZE   = 64
THRESHOLD_P  = 95


def load_data():
    data = np.load(DATA_PATH, allow_pickle=True)
    X    = data["X_raw"][..., np.newaxis]
    y    = (data["y_labels"] > 0).astype(np.int32)
    names = data["y_names"]
    return X, y, names


def get_mse(model, X):
    recon = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    return np.mean((X - recon) ** 2, axis=(1, 2))


def get_mse_tflite(path, X):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    mse_list = []
    for i in range(len(X)):
        s = X[i:i+1].astype(np.float32)
        interp.set_tensor(in_idx, s)
        interp.invoke()
        r = interp.get_tensor(out_idx).astype(np.float32)
        mse_list.append(float(np.mean((s - r) ** 2)))
    return np.array(mse_list)


def evaluate():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    X, y, names = load_data()

    print("Running Keras model...")
    model     = tf.keras.models.load_model(KERAS_MODEL)
    mse       = get_mse(model, X)

    # threshold = 95th percentile of normal sample MSE
    threshold = np.percentile(mse[y == 0], THRESHOLD_P)
    y_pred    = (mse > threshold).astype(np.int32)

    f1  = f1_score(y, y_pred, zero_division=0)
    pr  = precision_score(y, y_pred, zero_division=0)
    rc  = recall_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, mse)
    cm  = confusion_matrix(y, y_pred)

    print(f"\nF1:        {f1:.4f}")
    print(f"Precision: {pr:.4f}")
    print(f"Recall:    {rc:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    metrics = {"f1": f1, "precision": pr, "recall": rc,
               "auc_roc": auc, "threshold": float(threshold),
               "confusion_matrix": cm.tolist()}

    # tflite evaluation if model exists
    if os.path.exists(TFLITE_MODEL):
        print("\nRunning TFLite model...")
        mse_tfl   = get_mse_tflite(TFLITE_MODEL, X)
        thr_tfl   = np.percentile(mse_tfl[y == 0], THRESHOLD_P)
        y_pred_tfl = (mse_tfl > thr_tfl).astype(np.int32)
        metrics["tflite"] = {
            "f1"        : float(f1_score(y, y_pred_tfl, zero_division=0)),
            "auc_roc"   : float(roc_auc_score(y, mse_tfl)),
            "threshold" : float(thr_tfl)
        }
        print(f"TFLite F1: {metrics['tflite']['f1']:.4f}")

    # save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # MSE distribution plot
    plt.figure(figsize=(8, 4))
    plt.hist(mse[y == 0], bins=50, alpha=0.6, color="green", label="Normal", density=True)
    plt.hist(mse[y == 1], bins=50, alpha=0.6, color="red",   label="Fault",  density=True)
    plt.axvline(threshold, color="orange", linestyle="--", label=f"Threshold={threshold:.4f}")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Density")
    plt.title("MSE Distribution — Normal vs Fault")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mse_distribution.png"), dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, mse)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1],"k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # confusion matrix heatmap
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Fault"], yticklabels=["Normal","Fault"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    print("\nPlots saved to results/plots/")
    print("Metrics saved to results/metrics.json")


if __name__ == "__main__":
    evaluate()
