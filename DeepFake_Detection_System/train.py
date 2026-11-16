"""
AI Image Detector — Production Training Script
Classes: 0→REAL | 1→AI_GENERATED | 2→AI_EDITED
Architecture: EfficientNetB5 + Two-Stage Fine-Tuning
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16          # Lower for B5 (larger model)
EPOCHS_STAGE1   = 25
EPOCHS_STAGE2   = 35
LR_STAGE1       = 1e-4
LR_STAGE2       = 1e-5
UNFREEZE_N      = 120         # Unfreeze last 120 backbone layers in Stage 2
TTA_STEPS       = 5           # Test-Time Augmentation passes

CLASS_NAMES     = ["real", "ai_generated", "ai_edited"]
DATASET_DIR     = "dataset"
STATIC_DIR      = "static"
BEST_MODEL      = "best_model.keras"
FINAL_MODEL     = "final_model.keras"

os.makedirs(STATIC_DIR, exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE

# ─────────────────────────────────────────────
# MIXED PRECISION (GPU)
# ─────────────────────────────────────────────
if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed Precision enabled (GPU detected)")
else:
    print("ℹ️  Running on CPU")

# ─────────────────────────────────────────────
# AUGMENTATION PIPELINE (Strong)
# ─────────────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")


def preprocess(images, labels):
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images, labels


def augment_and_preprocess(images, labels):
    images = data_augmentation(images, training=True)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    return images, labels


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_dataset(split):
    path = os.path.join(DATASET_DIR, split)
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=42,
    )
    return ds


def build_pipeline(split):
    ds = load_dataset(split)
    if split == "train":
        ds = ds.map(augment_and_preprocess, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ─────────────────────────────────────────────
# CLASS WEIGHTS (Imbalance Handling)
# ─────────────────────────────────────────────
def compute_class_weights(train_ds):
    counts = np.zeros(len(CLASS_NAMES))
    for _, labels in train_ds:
        y = np.argmax(labels.numpy(), axis=1)
        for c in y:
            counts[c] += 1

    total = np.sum(counts)
    class_weights = {
        i: total / (len(CLASS_NAMES) * counts[i])
        for i in range(len(CLASS_NAMES))
    }

    print("\n📊 Class Distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {int(counts[i])}")
    print("\n⚖️  Class Weights:")
    for i, w in class_weights.items():
        print(f"  {CLASS_NAMES[i]} → {w:.4f}")
    return class_weights


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model():
    backbone = EfficientNetB5(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.6)(x)
    # dtype float32 for numerical stability
    outputs = layers.Dense(3, activation="softmax", dtype="float32")(x)

    model = Model(inputs, outputs)
    return model, backbone


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(stage):
    return [
        ModelCheckpoint(
            BEST_MODEL, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, verbose=1
        ),
        CSVLogger(f"training_stage{stage}.csv"),
    ]


# ─────────────────────────────────────────────
# TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────
def tta_predict(model, images, steps=TTA_STEPS):
    """
    Runs multiple augmented forward passes and averages the predictions.
    `images` should be a preprocessed batch (numpy or tensor).
    """
    preds = []
    for _ in range(steps):
        augmented = data_augmentation(images, training=True)
        p = model(augmented, training=False)
        preds.append(p.numpy())
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# PATCH-BASED PREDICTION
# ─────────────────────────────────────────────
def patch_predict(model, image_np, patch_size=112, stride=56):
    """
    Splits image into overlapping patches, predicts each,
    and returns the averaged probabilities.
    `image_np` — float32 array (H, W, 3), NOT yet preprocessed.
    """
    h, w = image_np.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image_np[y:y+patch_size, x:x+patch_size]
            patch = tf.image.resize(patch, IMG_SIZE).numpy()
            patches.append(patch)

    if not patches:
        patch = tf.image.resize(image_np, IMG_SIZE).numpy()
        patches.append(patch)

    patches = np.array(patches, dtype=np.float32)
    patches = tf.keras.applications.efficientnet.preprocess_input(patches)

    preds = model.predict(patches, verbose=0)
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# TRAINING CURVES PLOT
# ─────────────────────────────────────────────
def plot_training_curves(hist1, hist2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — AI Image Detector", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(hist1.history[f"val_{metric}"],   label="Stage 1 val",   color="#4A90D9")
        ax.plot(hist1.history[metric],             label="Stage 1 train", color="#4A90D9", linestyle="--", alpha=0.6)
        ax.plot(hist2.history[f"val_{metric}"],   label="Stage 2 val",   color="#E84393")
        ax.plot(hist2.history[metric],             label="Stage 2 train", color="#E84393", linestyle="--", alpha=0.6)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"✅ Training curves → {path}")


# ─────────────────────────────────────────────
# EVALUATION + CONFUSION MATRIX
# ─────────────────────────────────────────────
def evaluate_and_report(model, test_ds, use_tta=True):
    y_true, y_pred = [], []

    for images, labels in test_ds:
        if use_tta:
            preds = tta_predict(model, images.numpy())
        else:
            preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT" + (" (with TTA)" if use_tta else ""))
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix — Test Set"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"✅ Confusion matrix → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n🚀 Building data pipelines …")
    train_ds = build_pipeline("train")
    valid_ds = build_pipeline("valid")
    test_ds  = build_pipeline("test")

    class_weights = compute_class_weights(train_ds)

    # ── Stage 1: Frozen backbone ──────────────────────────────
    print("\n🔥 Stage 1: Training head (backbone frozen) …")
    model, backbone = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_STAGE1),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    hist1 = model.fit(
        train_ds, validation_data=valid_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=get_callbacks(stage=1),
        class_weight=class_weights,
    )

    # ── Stage 2: Fine-tune last N layers ─────────────────────
    print(f"\n🔥 Stage 2: Fine-tuning last {UNFREEZE_N} backbone layers …")
    backbone.trainable = True
    for layer in backbone.layers[:-UNFREEZE_N]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_STAGE2),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    hist2 = model.fit(
        train_ds, validation_data=valid_ds,
        epochs=EPOCHS_STAGE2,
        callbacks=get_callbacks(stage=2),
        class_weight=class_weights,
    )

    model.save(FINAL_MODEL)
    print(f"\n✅ Final model saved → {FINAL_MODEL}")

    plot_training_curves(hist1, hist2)

    best_model = tf.keras.models.load_model(BEST_MODEL)
    evaluate_and_report(best_model, test_ds, use_tta=True)

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()