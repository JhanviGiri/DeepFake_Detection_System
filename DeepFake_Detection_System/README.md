# DeepSight — AI Image Forensics System

A production-ready 3-class image forensics system using **EfficientNetB5**, **TTA**, **Patch-based Prediction**, and a **Flask** web app.

---

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | `REAL` | Unaltered photograph from a real camera |
| 1 | `AI_GENERATED` | Fully synthetic image (Stable Diffusion, MidJourney, DALL-E…) |
| 2 | `AI_EDITED` | Real photo manipulated with AI tools (inpainting, face swap…) |

---

## Project Structure

```
project/
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   ├── ai_generated/
│   │   └── ai_edited/
│   ├── valid/
│   └── test/
├── static/
│   └── uploads/          ← auto-created
├── templates/
│   └── index.html
├── prepare_dataset.py
├── train.py
├── test.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration:
```bash
pip install tensorflow[and-cuda]
```

### Step 2 — Prepare dataset

Edit the three path constants at the top of `prepare_dataset.py`:
```python
CIFAKE_DIR  = "path/to/CIFAKE"
CELEBA_DIR  = "path/to/img_align_celeba"
EDITED_DIR  = "path/to/training_fake"
```

Then run:
```bash
python prepare_dataset.py
```

This creates the `dataset/` folder with train/valid/test splits.

### Step 3 — Train

```bash
python train.py
```

**Outputs produced:**
| File | Description |
|------|-------------|
| `best_model.keras` | Best checkpoint (highest val accuracy) |
| `final_model.keras` | Weights after all fine-tuning epochs |
| `static/confusion_matrix.png` | Test-set confusion matrix |
| `static/training_curves.png` | Loss & accuracy plots |
| `training_stage1.csv` | Per-epoch log for Stage 1 |
| `training_stage2.csv` | Per-epoch log for Stage 2 |

### Step 4 — CLI Inference

```bash
python test.py --image path/to/image.jpg
```

Options:
```bash
python test.py --image photo.png --model best_model.keras
python test.py --image photo.png --no-tta       # disable Test-Time Augmentation
python test.py --image photo.png --no-patch     # disable patch-based prediction
```

Example output:
```
──────────────────────────────────────────────────────────
  Image     : photo.png
  Mode      : TTA + Patch
──────────────────────────────────────────────────────────
  Result    : AI_GENERATED
  Confidence: 97.32%

  Description:
  Fully synthetic image created by a generative model...

  Class Probabilities:
    REAL             [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.41%
    AI_GENERATED     [████████████████████████████░░] 97.32%  ◄
    AI_EDITED        [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  2.27%
──────────────────────────────────────────────────────────
```

### Step 5 — Web Application

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

Features:
- Drag & drop or click-to-browse
- Upload up to **5 images** at once
- Live spinner with inference step messages
- Grid layout showing all results
- Animated probability bars per class
- Full image preview

---

## Architecture

```
Input (224×224×3)
       │
EfficientNetB5 (ImageNet pretrained, frozen in Stage 1)
       │
GlobalAveragePooling2D
       │
BatchNormalization
       │
Dense(1024, relu)
       │
Dropout(0.6)
       │
Dense(3, softmax)  →  [P(REAL), P(AI_GEN), P(AI_EDITED)]
```

### Two-Stage Training

| Stage | Backbone | Learning Rate | Patience |
|-------|----------|---------------|---------|
| 1 — Head only | Fully frozen | 1e-4 | EarlyStopping(5), ReduceLR(3) |
| 2 — Fine-tune | Last 120 layers unfrozen | 1e-5 | EarlyStopping(5), ReduceLR(3) |

Both stages use **label smoothing (0.1)** and **class weighting**.

### Augmentation Pipeline (Training)

- `RandomFlip` (horizontal + vertical)
- `RandomRotation(0.2)`
- `RandomZoom(0.2)`
- `RandomContrast(0.2)`
- `RandomBrightness(0.15)`
- `RandomTranslation(0.1, 0.1)`

### Inference Enhancements

**Test-Time Augmentation (TTA):** Runs 5 augmented forward passes + 1 straight pass, then averages predictions.

**Patch-based Prediction:** Splits the image into overlapping 112×112 patches (stride 56), predicts each patch, and averages results — catches localized AI artifacts missed by whole-image analysis.

**Ensemble:** The final prediction is the average of TTA and patch probabilities.

---

## Production Deployment

```bash
# Install production WSGI server
pip install gunicorn

# Set a secure secret key
export FLASK_SECRET="your-very-long-random-string"
export MODEL_PATH="best_model.keras"

# Launch with 2 workers
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

---

## Tips for Higher Accuracy

1. **More data**: Scale up COUNTS in `prepare_dataset.py` (more data → better generalization)
2. **Larger input**: Change `IMG_SIZE` to `(380, 380)` for EfficientNetB5's native resolution
3. **CutMix / MixUp**: Add `tf_keras` augmentation callbacks for regularization
4. **Ensemble models**: Train multiple seeds and average their predictions

---

## Dataset Sources

| Class | Source |
|-------|--------|
| REAL | [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| AI_GENERATED | [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| AI_EDITED | [Real & Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) |