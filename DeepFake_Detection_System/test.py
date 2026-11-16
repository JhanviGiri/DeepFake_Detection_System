"""
test.py — CLI Inference for AI Image Detector
==============================================
Usage:
    python test.py --image path/to/image.jpg
    python test.py --image photo.png --model best_model.keras --no-tta --no-patch
"""

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG (must match train.py)
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
TTA_STEPS   = 5
CLASS_NAMES = ["REAL", "AI_GENERATED", "AI_EDITED"]
CLASS_META  = {
    "REAL":         "Unaltered photograph captured by a real camera.",
    "AI_GENERATED": "Fully synthetic image created by a generative model (Stable Diffusion, MidJourney, DALL-E…).",
    "AI_EDITED":    "Real photograph manipulated with AI tools (inpainting, face swap, generative fill…).",
}

# ─────────────────────────────────────────────
# AUGMENTATION (TTA)
# ─────────────────────────────────────────────
_tta_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])


def preprocess_input(arr: np.ndarray) -> np.ndarray:
    return tf.keras.applications.efficientnet.preprocess_input(arr.copy())


# ─────────────────────────────────────────────
# TTA PREDICT
# ─────────────────────────────────────────────
def tta_predict(model, img_arr: np.ndarray, steps=TTA_STEPS) -> np.ndarray:
    batch = np.expand_dims(img_arr, 0).astype(np.float32)
    preds = []
    for _ in range(steps):
        aug = _tta_aug(batch, training=True).numpy()
        p   = model(preprocess_input(aug), training=False).numpy()[0]
        preds.append(p)
    # Straight pass
    preds.append(model(preprocess_input(batch), training=False).numpy()[0])
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# PATCH PREDICT
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
        patches.append(tf.image.resize(img_arr, IMG_SIZE).numpy())
    patches = np.array(patches, dtype=np.float32)
    patches = preprocess_input(patches)
    preds   = model.predict(patches, verbose=0)
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, image_path: str, use_tta: bool, use_patch: bool) -> dict:
    img  = Image.open(image_path).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr  = np.array(img, dtype=np.float32)

    probs_list = []
    if use_tta:
        probs_list.append(tta_predict(model, arr))
    if use_patch:
        probs_list.append(patch_predict(model, arr))
    if not probs_list:
        batch = preprocess_input(np.expand_dims(arr, 0))
        probs_list.append(model.predict(batch, verbose=0)[0])

    probs      = np.mean(probs_list, axis=0)
    class_idx  = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[class_idx],
        "confidence":      float(probs[class_idx]),
        "probabilities":   dict(zip(CLASS_NAMES, probs.tolist())),
    }


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
def print_result(image_path: str, result: dict, use_tta: bool, use_patch: bool) -> None:
    BAR  = 30
    SEP  = "─" * 58
    flags = []
    if use_tta:   flags.append("TTA")
    if use_patch: flags.append("Patch")
    mode_str = f"  Mode      : {' + '.join(flags) if flags else 'Standard'}"

    print(SEP)
    print(f"  Image     : {image_path}")
    print(mode_str)
    print(SEP)
    print(f"  Result    : {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"\n  Description:")
    print(f"  {CLASS_META[result['predicted_class']]}")
    print(f"\n  Class Probabilities:")
    for name, prob in result["probabilities"].items():
        filled = int(prob * BAR)
        bar    = "█" * filled + "░" * (BAR - filled)
        marker = " ◄" if name == result["predicted_class"] else ""
        print(f"    {name:<16} [{bar}] {prob*100:5.2f}%{marker}")
    print(SEP + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Classify an image as Real, AI-Generated, or AI-Edited."
    )
    parser.add_argument("--image",   required=True,  help="Path to input image.")
    parser.add_argument("--model",   default="best_model.keras", help="Path to model file.")
    parser.add_argument("--no-tta",  action="store_true", help="Disable Test-Time Augmentation.")
    parser.add_argument("--no-patch",action="store_true", help="Disable patch-based prediction.")
    args = parser.parse_args()

    print(f"\n[Info] Loading model from: {args.model}")
    model  = tf.keras.models.load_model(args.model)
    print("[Info] Model loaded.\n")

    result = run_inference(
        model, args.image,
        use_tta   = not args.no_tta,
        use_patch = not args.no_patch,
    )
    print_result(args.image, result, not args.no_tta, not args.no_patch)


if __name__ == "__main__":
    main()