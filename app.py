
# app.py
import os
import glob
import json
import joblib
import threading
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model


MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
# optionally reduce TF log spam in container
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


app = Flask(__name__)


# locks for Keras predict (safe concurrency)
keras_lock = threading.Lock()


# --- utility: load classical artifacts ---
def load_classical():
scaler_path = os.path.join(MODELS_DIR, 'X_scaler.save')
le_path = os.path.join(MODELS_DIR, 'labelencoder.save')
rf_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
gb_path = os.path.join(MODELS_DIR, 'gb_model.joblib')


X_scaler = joblib.load(scaler_path)
le = joblib.load(le_path)
rf = joblib.load(rf_path)
gb = joblib.load(gb_path)
return X_scaler, le, rf, gb


# --- utility: find & load NN ensemble ---
def load_nn_ensemble():
meta_path = os.path.join(MODELS_DIR, 'nn_ensemble.meta.joblib')
model_files = []
y_mean = 0.0
y_std = 1.0
if os.path.exists(meta_path):
meta = joblib.load(meta_path)
y_mean = float(meta.get('y_mean', 0.0))
y_std = float(meta.get('y_std', 1.0))
for p in meta.get('model_paths', []):
full = os.path.join(MODELS_DIR, p)
if os.path.exists(full):
model_files.append(full)
# fallback: look for any .h5 in models dir
if not model_files:
model_files = sorted(glob.glob(os.path.join(MODELS_DIR, '*.h5')))
keras_models = []
for f in model_files:
try:
m = load_model(f, compile=False)
keras_models.append(m)
except Exception as e:
app.logger.warning(f"Failed to load Keras model {f}: {e}")
return keras_models, y_mean, y_std

# load at startup
X_scaler, label_enc, rf_model, gb_model = load_classical()
keras_models, y_mean, y_std = load_nn_ensemble()


@app.route('/health')
def health():
return jsonify({'status':'ok'})


@app.route('/predict', methods=['POST'])
def predict():
# Accept single object or list
payload = request.get_json(force=True)
if payload is None:
return jsonify({'error':'invalid json payload'}), 400


records = payload if isinstance(payload, list) else [payload]
# required keys
required = ['Red','IR','Gender','Age']
X_list = []
for i,rec in enumerate(records):
if not all(k in rec for k in required):
return jsonify({'error':f'record {i} missing one of {required}'}), 400
# encode gender using saved LabelEncoder
gender_raw = str(rec['Gender'])
if gender_raw not in list(label_enc.classes_):
return jsonify({'error':f'unknown Gender "{gender_raw}". Allowed: {list(label_enc.classes_)}'}), 400
gender_enc = int(label_enc.transform([gender_raw])[0])
X_list.append([float(rec['Red']), float(rec['IR']), float(gender_enc), float(rec['Age'])])


X = np.asarray(X_list, dtype=np.float32)
Xs = X_scaler.transform(X)


# classical preds
rf_pred = rf_model.predict(Xs)
gb_pred = gb_model.predict(Xs)


# NN preds (average across loaded models)
nn_mean = None
if len(keras_models) > 0:
all_preds = []
with keras_lock:
for m in keras_models:
p = m.predict(Xs).reshape(-1)
# if model was trained scaled, meta contains mean/std; convert if needed
p = p * y_std + y_mean
all_preds.append(p)
nn_mean = np.mean(np.stack(all_preds, axis=0), axis=0)


# ensemble: average available model predictions
parts = [rf_pred, gb_pred]
if nn_mean is not None:
parts.append(nn_mean)
ensemble = np.mean(np.vstack(parts), axis=0)


out = []
for i in range(len(ensemble)):
entry = {
'rf': float(rf_pred[i]),
'gb': float(gb_pred[i]),
'prediction': float(ensemble[i])
}
if nn_mean is not None:
entry['nn'] = float(nn_mean[i])
out.append(entry)


return jsonify({'results': out})


if __name__ == '__main__':
# local dev
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
