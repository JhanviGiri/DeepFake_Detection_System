"""
app.py — Production Flask Backend for AI Image Detector
Supports: Multi-image upload (max 5), TTA, Patch-based prediction
"""

import os
import io
import base64
import uuid
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE       = (224, 224)
MAX_IMAGES     = 5
MAX_FILE_MB    = 16
ALLOWED_EXT    = {"jpg", "jpeg", "png", "webp"}
MODEL_PATH     = os.environ.get("MODEL_PATH", "best_model.keras")
TTA_STEPS      = 5
USE_TTA        = True
USE_PATCH      = True

CLASS_NAMES = ["REAL", "AI_GENERATED", "AI_EDITED"]
CLASS_META  = {
    "REAL": {
        "icon": "📷",
        "color": "#22c55e",
        "label": "Authentic Photo",
        "description": "This image shows characteristics of a genuine photograph captured by a real camera. No signs of AI generation or significant AI-based manipulation were detected.",
    },
    "AI_GENERATED": {
        "icon": "🤖",
        "color": "#f97316",
        "label": "AI Generated",
        "description": "This image appears to be fully synthetic — created entirely by a generative AI model such as Stable Diffusion, MidJourney, or DALL-E.",
    },
    "AI_EDITED": {
        "icon": "✏️",
        "color": "#a855f7",
        "label": "AI Edited",
        "description": "This image appears to be a real photograph that has been manipulated using AI tools, such as inpainting, face swapping, or generative fill.",
    },
}

app = Flask(__name__)
app.config["SECRET_KEY"]       = os.environ.get("FLASK_SECRET", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# MODEL LOADING (lazy, cached)
# ─────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Please run train.py first, then restart the app."
            )
        print(f"[INFO] Loading model from {MODEL_PATH} …")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model ready.")
    return _model


# ─────────────────────────────────────────────
# AUGMENTATION (for TTA)
# ─────────────────────────────────────────────
_tta_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def preprocess_batch(arr_batch: np.ndarray) -> np.ndarray:
    return tf.keras.applications.efficientnet.preprocess_input(arr_batch.copy())


# ─────────────────────────────────────────────
# TTA PREDICT
# ─────────────────────────────────────────────
def tta_predict(model, img_arr: np.ndarray, steps=TTA_STEPS) -> np.ndarray:
    batch = np.expand_dims(img_arr, 0).astype(np.float32)  # (1,H,W,3)
    preds = []
    for _ in range(steps):
        aug  = _tta_aug(batch, training=True).numpy()
        proc = preprocess_batch(aug)
        p    = model(proc, training=False).numpy()
        preds.append(p[0])
    # Also add the straight (no-aug) pass
    proc = preprocess_batch(batch)
    preds.append(model(proc, training=False).numpy()[0])
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# PATCH-BASED PREDICT
# ─────────────────────────────────────────────
def patch_predict(model, img_arr: np.ndarray, patch_size=112, stride=56) -> np.ndarray:
    h, w = img_arr.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_arr[y:y+patch_size, x:x+patch_size]
            patch = tf.image.resize(patch, IMG_SIZE).numpy()
            patches.append(patch)

    if not patches:
        patches.append(img_arr)

    patches = np.array(patches, dtype=np.float32)
    patches = preprocess_batch(patches)
    preds   = model.predict(patches, verbose=0)
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# COMBINED INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, img_arr: np.ndarray) -> dict:
    """
    Combines TTA + patch-based prediction for maximum accuracy.
    """
    probs_list = []

    if USE_TTA:
        probs_list.append(tta_predict(model, img_arr))

    if USE_PATCH:
        probs_list.append(patch_predict(model, img_arr))

    if not probs_list:
        batch = preprocess_batch(np.expand_dims(img_arr, 0))
        p     = model.predict(batch, verbose=0)[0]
        probs_list.append(p)

    probs      = np.mean(probs_list, axis=0)
    class_idx  = int(np.argmax(probs))
    class_name = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])

    return {
        "predicted_class": class_name,
        "class_index":     class_idx,
        "confidence":      round(confidence * 100, 2),
        "probabilities":   {
            name: round(float(p) * 100, 2)
            for name, p in zip(CLASS_NAMES, probs)
        },
        "meta": CLASS_META[class_name],
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    model_ready = os.path.exists(MODEL_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded."}), 400

    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Maximum {MAX_IMAGES} images allowed per request."}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    results = []

    for file in files:
        if not file or file.filename == "":
            continue
        if not allowed_file(file.filename):
            results.append({"filename": file.filename, "error": "Unsupported file type."})
            continue

        try:
            pil_img  = Image.open(file.stream)
            img_arr  = pil_to_array(pil_img)
            result   = run_inference(model, img_arr)
            thumb_b64 = image_to_base64(pil_img.resize((320, 320), Image.LANCZOS))

            results.append({
                "filename":   secure_filename(file.filename),
                "thumbnail":  thumb_b64,
                **result,
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify({"results": results})


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)