import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, roc_auc_score

# Get absolute path to processed data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/cwru_processed.npz")

def test_uno_pipeline():
    # 1. Load data
    data = np.load(DATA_PATH, allow_pickle=True)
    X_feat = data["X_features"]
    y = data["y_labels"]
    
    # Slice first 7 features (time-domain stats)
    X_time = X_feat[:, :7]
    print(f"Features shape: {X_time.shape}")
    
    # Split normal data
    X_normal = X_time[y == 0]
    X_fault = X_time[y != 0]
    y_fault = y[y != 0]
    
    n_train = int(len(X_normal) * 0.8)
    idx = np.random.permutation(len(X_normal))
    X_train = X_normal[idx[:n_train]]
    X_val = X_normal[idx[n_train:]]
    
    # 2. Build tiny model
    inputs = keras.Input(shape=(7,))
    x = layers.Dense(8, activation="relu")(inputs)
    bottleneck = layers.Dense(3, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(bottleneck)
    outputs = layers.Dense(7, activation="linear")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Uno_Tiny_AE")
    model.compile(optimizer="adam", loss="mse")
    
    # 3. Train
    print("Training model...")
    model.fit(X_train, X_train, validation_data=(X_val, X_val), 
              batch_size=32, epochs=30, verbose=0)
    
    # 4. Threshold & Eval
    r_val = model.predict(X_val, verbose=0)
    mse_val = np.mean((X_val - r_val) ** 2, axis=1)
    threshold = np.percentile(mse_val, 95)
    print(f"Threshold: {threshold:.6f}")
    
    # Test all
    r_all = model.predict(X_time, verbose=0)
    mse_all = np.mean((X_time - r_all) ** 2, axis=1)
    
    y_pred = (mse_all > threshold).astype(int)
    y_true = (y != 0).astype(int)
    
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, mse_all)
    print(f"F1-Score: {f1:.4f}  |  AUC-ROC: {auc:.4f}")
    
    # 5. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = converter.convert()
    print(f"Float32 TFLite Size: {len(tflite_f32)} bytes ({len(tflite_f32)/1024:.2f} KB)")
    
    # Quantized TFLite
    converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    def rep_gen():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]
    converter_q.representative_dataset = rep_gen
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_q.inference_input_type = tf.float32
    converter_q.inference_output_type = tf.float32
    tflite_int8 = converter_q.convert()
    print(f"Int8 TFLite Size: {len(tflite_int8)} bytes ({len(tflite_int8)/1024:.2f} KB)")

if __name__ == "__main__":
    test_uno_pipeline()
