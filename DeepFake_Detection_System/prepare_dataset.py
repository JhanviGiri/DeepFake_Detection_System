"""
prepare_dataset.py — Dataset preparation for AI Image Detector
===============================================================
Sources:
  • REAL        → CelebA aligned face images
  • AI_GENERATED → CIFAKE FAKE folder (train + test)
  • AI_EDITED   → Real & Fake Face Detection training_fake folder

Edit the three path constants below, then run:
    python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

# ─── EDIT THESE PATHS ────────────────────────────────────────────────────────
CIFAKE_DIR  = r"C:\Users\DELL\Downloads\CIFAKE"
CELEBA_DIR  = r"C:\Users\DELL\Downloads\CelebA\img_align_celeba\img_align_celeba"
EDITED_DIR  = r"C:\Users\DELL\Downloads\Real & Fake Face Detection\real_and_fake_face\training_fake"
# ─────────────────────────────────────────────────────────────────────────────

COUNTS = {
    "train": {"real": 600, "ai_generated": 600, "ai_edited": 600},
    "valid": {"real": 150, "ai_generated": 150, "ai_edited": 150},
    "test":  {"real": 150, "ai_generated": 150, "ai_edited": 210},
}

SPLITS  = ["train", "valid", "test"]
CLASSES = ["real", "ai_generated", "ai_edited"]

random.seed(42)


def get_image_files(folder: str) -> list:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in extensions]
    random.shuffle(files)
    return files


def copy_images(src_list: list, dst_folder: str, count: int, prefix: str = "") -> list:
    os.makedirs(dst_folder, exist_ok=True)
    taken = src_list[:count]
    for i, src in enumerate(taken):
        ext = Path(src).suffix
        dst = os.path.join(dst_folder, f"{prefix}_{i:06d}{ext}")
        shutil.copy2(src, dst)
    print(f"  ✓ Copied {len(taken)} images → {dst_folder}")
    return src_list[count:]


def main():
    print("\n=== Dataset Preparation ===\n")

    real_imgs   = get_image_files(CELEBA_DIR)
    gen_imgs    = (
        get_image_files(os.path.join(CIFAKE_DIR, "train", "FAKE")) +
        get_image_files(os.path.join(CIFAKE_DIR, "test",  "FAKE"))
    )
    random.shuffle(gen_imgs)
    edited_imgs = get_image_files(EDITED_DIR)

    total_needed = {cls: sum(COUNTS[s][cls] for s in SPLITS) for cls in CLASSES}

    print(f"CelebA found   : {len(real_imgs):,}   (need {total_needed['real']:,})")
    print(f"CIFAKE FAKE    : {len(gen_imgs):,}   (need {total_needed['ai_generated']:,})")
    print(f"Edited FAKE    : {len(edited_imgs):,}   (need {total_needed['ai_edited']:,})")

    checks = [
        (real_imgs,   total_needed["real"],          "CelebA"),
        (gen_imgs,    total_needed["ai_generated"],  "CIFAKE"),
        (edited_imgs, total_needed["ai_edited"],     "Edited"),
    ]
    for pool, needed, label in checks:
        if len(pool) < needed:
            print(f"\n✗ Not enough {label} images. Have {len(pool)}, need {needed}.")
            return

    for split in SPLITS:
        print(f"\n[{split.upper()}]")
        real_imgs   = copy_images(real_imgs,   f"dataset/{split}/real",         COUNTS[split]["real"],         "real")
        gen_imgs    = copy_images(gen_imgs,    f"dataset/{split}/ai_generated",  COUNTS[split]["ai_generated"], "gen")
        edited_imgs = copy_images(edited_imgs, f"dataset/{split}/ai_edited",     COUNTS[split]["ai_edited"],    "edit")

    print("\n✅ Dataset ready inside 'dataset/' folder.")
    print("   Next step: python train.py")


if __name__ == "__main__":
    main()