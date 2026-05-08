"""
Builds augmented test sets by importing transform functions from augment_testset.py.

Edit the CONFIGURATION section below to control:
  - which transforms to run
  - how many images per class to sample
  - where input and output live
  - the random seed

Each run produces one directory under OUTPUT_ROOT, mirroring the original
test structure:
    domain-shift/
    └── augmented_test/
        ├── test_blur/
        │   ├── bent/                  ← n augmented images
        │   ├── color/
        │   ├── flip/
        │   ├── good/
        │   ├── scratch/
        │   └── augmentation_log.json  ← one file documenting all params
        ├── test_exposure/
        └── ...

Run:
    python build_augmented_sets.py
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

# Import all transform functions from the sibling file
from domain_shift_functions import (
    apply_blur,
    apply_contrast,
    apply_exposure,
    apply_gamma,
    apply_jpeg,
    apply_noise,
    apply_perspective,
    apply_affine,
    apply_shadow,
    apply_specular,
    apply_vignette,
    apply_white_balance,
    CLASS_FOLDERS,
    VALID_EXTS,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this section
# ──────────────────────────────────────────────────────────────────────────────

# Root of the project (parent of domain-shift/)
ROOT = Path(__file__).parent.parent

# Input: original MVTec test directory
INPUT_DIR = ROOT / "data" / "metal_nut" / "test"

# Output: all augmented sets go here, original data is never touched
OUTPUT_ROOT = Path(__file__).parent / "metal_nut_augmented" / "test"

# Random seed — controls which images are sampled, not the augmentation params
# (augmentation params have their own seed = SEED + 1)
SEED = 42

# Number of images to sample per class per run.
# Set to None to use all available images.
N_IMAGES = 20

# Runs to generate. Each entry is:
#   "output_directory_name": [list of transform functions to apply in order]
#
# If you list more than one function they are applied sequentially to the same image.
# The image sampled is always random (reproducible via SEED), the transforms
# are applied deterministically given the augmentation seed.
#
# To test a single transform in isolation, put only one function in the list.
# To test a combination, put multiple — they run left to right.

RUNS = {
    "test_exposure":    [apply_exposure],
    #"test_gamma":       [apply_gamma],
    #"test_wb":          [apply_white_balance],
    #"test_noise":       [apply_noise],
    #"test_jpeg":        [apply_jpeg],
    #"test_blur":        [apply_blur],
    #"test_vignette":    [apply_vignette],
    #"test_shadow":      [apply_shadow],
    #"test_contrast":    [apply_contrast],
    #"test_affine":      [apply_affine],
    #"test_perspective": [apply_perspective],
    #"test_specular":    [apply_specular],

    # Example combinations 
    #"test_photometric": [apply_exposure, apply_white_balance, apply_noise],
    #"test_geometric":   [apply_affine,   apply_perspective],
    #"test_optical":     [apply_blur,     apply_vignette,     apply_jpeg],
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers 
# ──────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def _save_img(img: np.ndarray, path: Path) -> None:
    """Save image only - params are collected into the run-level JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _sample(class_dir: Path, n, rng: random.Random) -> list[Path]:
    """Return n randomly sampled image paths from class_dir."""
    imgs = sorted(p for p in class_dir.iterdir()
                  if p.suffix.lower() in VALID_EXTS)
    if not imgs:
        return []
    if n is None or n >= len(imgs):
        return imgs
    return rng.sample(imgs, n)


def _apply_chain(img: np.ndarray,
                 fns: list,
                 aug_rng: random.Random) -> tuple[np.ndarray, dict]:
    """Apply a list of transform functions in sequence, collecting all params."""
    out        = img.copy()
    fn_names   = [fn.__name__.replace("apply_", "") for fn in fns]
    all_params = {"transforms_applied": fn_names}

    for fn in fns:
        out, p = fn(out, aug_rng)
        name   = fn.__name__.replace("apply_", "")
        all_params[name] = p

    return out, all_params


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}\n"
            f"Check the ROOT path in the CONFIGURATION section."
        )

    # Two separate seeded RNGs so that changing N_IMAGES never
    # affects which augmentation parameters are drawn, and vice versa.
    sample_rng = random.Random(SEED)
    aug_rng    = random.Random(SEED + 1)
    np.random.seed(SEED)

    print(f"\nInput  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_ROOT}")
    print(f"Seed   : {SEED}   n_images/class: {N_IMAGES or 'all'}")
    print(f"Runs   : {len(RUNS)}\n")

    for run_name, transform_fns in RUNS.items():
        out_dir  = OUTPUT_ROOT / run_name
        fn_names = [fn.__name__ for fn in transform_fns]
        run_log  = {}   # { "class/filename": params_dict }
        total    = 0

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{run_name}]  transforms: {fn_names}")

        for cls in CLASS_FOLDERS:
            class_dir     = INPUT_DIR / cls
            out_class_dir = out_dir / cls

            if not class_dir.exists():
                print(f"    [!] {cls} not found — skipping.")
                continue

            images = _sample(class_dir, N_IMAGES, sample_rng)
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for src in images:
                img         = _load(src)
                aug, params = _apply_chain(img, transform_fns, aug_rng)
                _save_img(aug, out_class_dir / src.name)
                run_log[f"{cls}/{src.name}"] = params

            total += len(images)
            print(f"    {cls:12s} {len(images):3d} images")

        # One JSON per run documenting every image and its exact parameters
        log_entry = {
            "run_name":     run_name,
            "transforms":   fn_names,
            "n_per_class":  N_IMAGES,
            "seed":         SEED,
            "input_dir":    str(INPUT_DIR),
            "output_dir":   str(out_dir),
            "total_images": total,
            "images":       run_log,
        }
        (out_dir / "augmentation_log.json").write_text(
            json.dumps(log_entry, indent=2)
        )

        print(f"    {'total':12s} {total:3d} images → {out_dir}")
        print(f"    log      → augmentation_log.json\n")
    print("Done.")


if __name__ == "__main__":
    main()