import os
import io
import base64
import json
import time
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, '../models/cae_quantized.tflite')
DATA_PATH   = os.path.join(BASE_DIR, '../data/processed/cwru_processed.npz')
SCALER_MEAN = os.path.join(BASE_DIR, '../data/processed/cwru_scaler_mean.npy')
SCALER_STD  = os.path.join(BASE_DIR, '../data/processed/cwru_scaler_scale.npy')
CWRU_DIR    = os.path.join(BASE_DIR, '../CWRU/raw')

WINDOW_SIZE = 2048
SR          = 48000

FILE_MAP = {
    'Normal'        : 'Time_Normal_1_098.mat',
    'Ball_007'      : 'B007_1_123.mat',
    'Ball_014'      : 'B014_1_190.mat',
    'Ball_021'      : 'B021_1_227.mat',
    'InnerRace_007' : 'IR007_1_110.mat',
    'InnerRace_014' : 'IR014_1_175.mat',
    'InnerRace_021' : 'IR021_1_214.mat',
    'OuterRace_007' : 'OR007_6_1_136.mat',
    'OuterRace_014' : 'OR014_6_1_202.mat',
    'OuterRace_021' : 'OR021_6_1_239.mat',
}

# load model and scaler once at startup
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
IN_IDX  = interp.get_input_details()[0]['index']
OUT_IDX = interp.get_output_details()[0]['index']

scaler_mean  = np.load(SCALER_MEAN)
scaler_scale = np.load(SCALER_STD)

# compute threshold from normal data
data   = np.load(DATA_PATH, allow_pickle=True)
X_all  = data['X_raw'][..., np.newaxis].astype(np.float32)
y      = data['y_labels']

mse_n = []
for i in np.where(y == 0)[0]:
    s = X_all[i:i+1]
    interp.set_tensor(IN_IDX, s)
    interp.invoke()
    r = interp.get_tensor(OUT_IDX)
    mse_n.append(float(np.mean((s - r)**2)))

THRESHOLD = float(np.percentile(mse_n, 95))
print(f'App ready  |  Threshold = {THRESHOLD:.6f}')


def load_mat_signal(filepath):
    mat = sio.loadmat(filepath)
    for k in mat.keys():
        if 'DE_time' in k:
            return mat[k].flatten().astype(np.float32)
    for k in mat.keys():
        if 'time' in k.lower() and not k.startswith('__'):
            return mat[k].flatten().astype(np.float32)
    return None


def run_inference(window_raw):
    norm = (window_raw - scaler_mean) / scaler_scale
    x    = norm[np.newaxis, :, np.newaxis].astype(np.float32)
    t0   = time.perf_counter()
    interp.set_tensor(IN_IDX, x)
    interp.invoke()
    r    = interp.get_tensor(OUT_IDX)
    lat  = (time.perf_counter() - t0) * 1000
    score = float(np.mean((x - r)**2))
    return score, lat, x[0, :, 0], r[0, :, 0]


def make_plot(original, reconstructed, score, threshold, label):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), facecolor='white')
    color = '#d32f2f' if score > threshold else '#388e3c'

    for ax in axes:
        ax.set_facecolor('white')
        ax.tick_params(colors='#333')
        ax.spines['bottom'].set_color('#ccc')
        ax.spines['top'].set_color('#ccc')
        ax.spines['left'].set_color('#ccc')
        ax.spines['right'].set_color('#ccc')

    axes[0].plot(original,      linewidth=0.7, color='#1976d2', label='Input')
    axes[0].plot(reconstructed, linewidth=0.7, color=color, linestyle='--', label='Reconstruction')
    axes[0].set_title(f'{label}  |  MSE = {score:.6f}', color='black', fontsize=10)
    axes[0].set_ylabel('Amplitude', color='#333')
    axes[0].legend(fontsize=8, facecolor='white', labelcolor='black')
    axes[0].grid(True, alpha=0.3, color='#ddd')

    err = np.abs(original - reconstructed)
    axes[1].fill_between(range(len(err)), err, color=color, alpha=0.4)
    axes[1].axhline(np.sqrt(threshold), color='orange', linestyle='--',
                     linewidth=1.5, label='Threshold')
    axes[1].set_ylabel('|Error|', color='#333')
    axes[1].set_xlabel('Sample', color='#333')
    axes[1].set_title('Reconstruction Error', color='black', fontsize=10)
    axes[1].legend(fontsize=8, facecolor='white', labelcolor='black')
    axes[1].grid(True, alpha=0.3, color='#ddd')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_sliding_plot(signal_raw, label):
    STEP      = 1024
    N_WINDOWS = 40
    scores, flags = [], []

    for i in range(N_WINDOWS):
        start  = i * STEP
        window = signal_raw[start:start + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            break
        score, _, _, _ = run_inference(window)
        scores.append(score)
        flags.append(score > THRESHOLD)

    n = len(scores)
    win_times = [(i * STEP + WINDOW_SIZE//2) / SR * 1000 for i in range(n)]
    colors    = ['#d32f2f' if f else '#388e3c' for f in flags]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')
        ax.tick_params(colors='#333')
        for s in ax.spines.values():
            s.set_color('#ccc')

    t_ms = np.arange(min(n * STEP + WINDOW_SIZE, len(signal_raw))) / SR * 1000
    axes[0].plot(t_ms, signal_raw[:len(t_ms)], linewidth=0.5, color='#1976d2')
    axes[0].set_ylabel('Amplitude', color='#333')
    axes[0].set_title(f'Signal: {label}', color='black', fontsize=10)
    axes[0].grid(True, alpha=0.3, color='#ddd')

    axes[1].bar(win_times, scores,
                width=STEP/SR*1000*0.8, color=colors, alpha=0.8)
    axes[1].axhline(THRESHOLD, color='orange', linestyle='--',
                     linewidth=1.5, label=f'Threshold={THRESHOLD:.5f}')
    axes[1].set_xlabel('Time (ms)', color='#333')
    axes[1].set_ylabel('Anomaly Score', color='#333')
    axes[1].set_title('Sliding Window Detection', color='black', fontsize=10)
    axes[1].legend(fontsize=8, facecolor='white', labelcolor='black')
    axes[1].grid(True, alpha=0.3, color='#ddd')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), sum(flags), n


# ── Routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    classes = list(FILE_MAP.keys())
    return render_template('index.html',
                           classes=classes,
                           threshold=round(THRESHOLD, 6))


@app.route('/predict_class', methods=['POST'])
def predict_class():
    cls  = request.json.get('class_name', 'Normal')
    fname = FILE_MAP.get(cls)
    fpath = os.path.join(CWRU_DIR, fname)

    if not os.path.exists(fpath):
        return jsonify({'error': f'File not found: {fname}'}), 404

    signal = load_mat_signal(fpath)
    mid    = len(signal)//2
    window = signal[mid:mid + WINDOW_SIZE]

    score, latency, orig, recon = run_inference(window)
    is_fault = score > THRESHOLD
    plot_b64 = make_plot(orig, recon, score, THRESHOLD, cls)
    _, n_fault, n_total = make_sliding_plot(signal, cls)
    slide_b64, _, _ = make_sliding_plot(signal, cls)

    return jsonify({
        'score'    : round(score, 6),
        'threshold': round(THRESHOLD, 6),
        'latency'  : round(latency, 2),
        'is_fault' : is_fault,
        'class'    : cls,
        'plot'     : plot_b64,
        'slide'    : slide_b64,
        'n_fault'  : n_fault,
        'n_total'  : n_total,
    })


@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f    = request.files['file']
    data_bytes = f.read()
    tmp  = os.path.join(BASE_DIR, '_tmp_upload.mat')
    with open(tmp, 'wb') as out:
        out.write(data_bytes)

    signal = load_mat_signal(tmp)
    os.remove(tmp)

    if signal is None:
        return jsonify({'error': 'Could not read signal from file'}), 400

    mid    = len(signal)//2
    window = signal[mid:mid + WINDOW_SIZE]

    score, latency, orig, recon = run_inference(window)
    is_fault = score > THRESHOLD
    plot_b64 = make_plot(orig, recon, score, THRESHOLD, f.filename)
    slide_b64, n_fault, n_total = make_sliding_plot(signal, f.filename)

    return jsonify({
        'score'    : round(score, 6),
        'threshold': round(THRESHOLD, 6),
        'latency'  : round(latency, 2),
        'is_fault' : is_fault,
        'class'    : f.filename,
        'plot'     : plot_b64,
        'slide'    : slide_b64,
        'n_fault'  : n_fault,
        'n_total'  : n_total,
    })

@app.route('/stream_data', methods=['POST'])
def stream_data():
    cls = request.json.get('class_name', 'Normal')
    idx = request.json.get('index', 0)
    
    fname = FILE_MAP.get(cls)
    fpath = os.path.join(CWRU_DIR, fname)
    if not os.path.exists(fpath):
        return jsonify({'error': f'File not found: {fname}'}), 404
        
    signal = load_mat_signal(fpath)
    
    step = 1024
    start = idx * step
    if start + WINDOW_SIZE > len(signal):
        start = 0
        idx = 0
        
    window = signal[start:start + WINDOW_SIZE]
    
    score, latency, orig, recon = run_inference(window)
    is_fault = score > THRESHOLD
    
    return jsonify({
        'index': idx,
        'score': round(score, 6),
        'threshold': round(THRESHOLD, 6),
        'is_fault': is_fault,
        'latency': round(latency, 2),
        'original': orig.tolist()[:200],
        'reconstructed': recon.tolist()[:200]
    })


@app.route('/all_classes')
def all_classes():
    results = []
    for cls, fname in FILE_MAP.items():
        fpath = os.path.join(CWRU_DIR, fname)
        if not os.path.exists(fpath):
            continue
        signal = load_mat_signal(fpath)
        mid    = len(signal)//2
        window = signal[mid:mid + WINDOW_SIZE]
        score, latency, _, _ = run_inference(window)
        results.append({
            'class'    : cls,
            'score'    : round(score, 6),
            'latency'  : round(latency, 2),
            'is_fault' : score > THRESHOLD,
            'expected' : cls != 'Normal',
            'correct'  : (score > THRESHOLD) == (cls != 'Normal'),
        })
    return jsonify({'results': results, 'threshold': round(THRESHOLD, 6)})


if __name__ == '__main__':
    print('Starting Flask app at http://127.0.0.1:5000')
    app.run(debug=False, host='127.0.0.1', port=5000)
