import os
import time
import json
import numpy as np
import tensorflow as tf


# paths
KERAS_MODEL   = r"../models/cae_model.h5"
PRUNED_MODEL  = r"../models/cae_pruned.h5"
TFLITE_F32    = r"../models/cae_float32.tflite"
TFLITE_INT8   = r"../models/cae_quantized.tflite"
DATA_PATH     = r"../data/processed/cwru_processed.npz"
RESULTS_DIR   = r"../results"

N_CALIB  = 200   # samples for int8 calibration
N_RUNS   = 100   # benchmark runs


def get_calib_data(n=200):
    data   = np.load(DATA_PATH, allow_pickle=True)
    X      = data["X_raw"]
    y      = data["y_labels"]
    normal = X[y == 0][:n][..., np.newaxis].astype(np.float32)
    return normal


def to_tflite_f32(keras_path, out_path):
    model = tf.keras.models.load_model(keras_path)
    conv  = tf.lite.TFLiteConverter.from_keras_model(model)
    fb    = conv.convert()
    with open(out_path, "wb") as f:
        f.write(fb)
    print(f"float32 TFLite: {len(fb)/1024:.1f} KB -> {out_path}")
    return len(fb)


def to_tflite_int8(keras_path, calib_data, out_path):
    model = tf.keras.models.load_model(keras_path)
    conv  = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for i in range(len(calib_data)):
            yield [calib_data[i:i+1]]

    conv.representative_dataset    = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.float32
    conv.inference_output_type     = tf.float32

    fb = conv.convert()
    with open(out_path, "wb") as f:
        f.write(fb)
    print(f"int8 TFLite:    {len(fb)/1024:.1f} KB -> {out_path}")
    return len(fb)


def prune_model(keras_path, pruned_path, calib_data, sparsity=0.5):
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        print("tensorflow-model-optimization not installed, skipping pruning")
        return False

    model = tf.keras.models.load_model(keras_path)
    params = {"pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(sparsity, begin_step=0)}
    pruned = tfmot.sparsity.keras.prune_low_magnitude(model, **params)
    pruned.compile(optimizer="adam", loss="mse")

    # brief fine-tune after pruning
    pruned.fit(calib_data, calib_data, batch_size=64, epochs=3, verbose=1,
               callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

    stripped = tfmot.sparsity.keras.strip_pruning(pruned)
    stripped.save(pruned_path)
    print(f"Pruned model saved: {pruned_path}")
    return True


def benchmark(tflite_path, sample, n=100):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]

    # warmup
    interp.set_tensor(in_idx, sample)
    interp.invoke()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        interp.set_tensor(in_idx, sample)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def main():
    os.makedirs(os.path.dirname(TFLITE_F32), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading calibration data...")
    calib = get_calib_data(N_CALIB)

    # prune (optional)
    pruned_ok = prune_model(KERAS_MODEL, PRUNED_MODEL, calib)
    source = PRUNED_MODEL if pruned_ok else KERAS_MODEL

    # convert
    print("\nConverting to TFLite...")
    f32_bytes = to_tflite_f32(KERAS_MODEL, TFLITE_F32)
    i8_bytes  = to_tflite_int8(source, calib, TFLITE_INT8)

    # benchmark
    print(f"\nBenchmarking ({N_RUNS} runs)...")
    sample  = calib[0:1]
    f32_ms, f32_std = benchmark(TFLITE_F32, sample, N_RUNS)
    i8_ms,  i8_std  = benchmark(TFLITE_INT8, sample, N_RUNS)

    keras_kb  = os.path.getsize(KERAS_MODEL) / 1024
    pruned_kb = os.path.getsize(PRUNED_MODEL) / 1024 if os.path.exists(PRUNED_MODEL) else 0

    print("\n--- Results ---")
    print(f"Keras H5          : {keras_kb:.0f} KB")
    if pruned_kb:
        print(f"Pruned H5         : {pruned_kb:.0f} KB")
    print(f"TFLite float32    : {f32_bytes/1024:.0f} KB  |  {f32_ms:.2f} ms")
    print(f"TFLite int8       : {i8_bytes/1024:.0f} KB  |  {i8_ms:.2f} ms")
    print(f"Compression ratio : {keras_kb/(i8_bytes/1024):.1f}x")

    result = {
        "keras_kb"    : keras_kb,
        "pruned_kb"   : pruned_kb,
        "f32_kb"      : f32_bytes / 1024,
        "int8_kb"     : i8_bytes  / 1024,
        "f32_ms"      : f32_ms,
        "int8_ms"     : i8_ms,
        "compression" : keras_kb / (i8_bytes / 1024)
    }
    with open(os.path.join(RESULTS_DIR, "benchmark.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved benchmark.json")


if __name__ == "__main__":
    main()
