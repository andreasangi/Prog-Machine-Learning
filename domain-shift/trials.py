import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


CLASS_FOLDERS = ["bent", "color", "flip", "good", "scratch"]
VALID_EXTS = {".png", ".jpg", ".jpeg"}


def _clip(img: np.ndarray) -> np.ndarray:
    """Clip float image to [0, 255] and cast to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def collect_test_images(test_root: Path) -> list[Path]:
    """Collect images from MVTec metal_nut test subfolders."""
    paths: list[Path] = []
    for cls in CLASS_FOLDERS:
        cls_dir = test_root / cls
        if not cls_dir.exists():
            continue
        cls_paths = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ]
        paths.extend(sorted(cls_paths))
    return paths


def choose_samples(paths: list[Path], n_samples: int, rng: random.Random, random_pick: bool) -> list[Path]:
    if len(paths) == 0:
        return []
    n = min(n_samples, len(paths))
    if random_pick:
        return rng.sample(paths, n)
    return paths[:n]

def apply_white_balance(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Per-channel multiplicative drift to simulate white-balance miscalibration.

    Each BGR channel is not scaled independently since Real illuminant shifts 
    mainly affect the R/B ratio while G stays relatively stable.  Shifts are small and
    asymmetric so the overall cast is realistic (not rainbow-coloured).

    Industrial case: switching between fluorescent, LED, halogen and sodium-
    vapour lighting changes the illuminant spectrum; a fixed white-balance
    preset introduces a colour cast.
    Plausible range: per-channel scale ∈ [0.75, 1.25]
    """
    scale_G = rng.uniform(0.90, 1.10)          # subtle
    scale_R = rng.uniform(0.70, 1.40)          # warm vs cool shift
    scale_B = rng.uniform(0.70, 1.40)          # correlated opposite to R

    # for a more realistic effect we anti-correlate R and B to model colour temperature shift
    # (warm light = more R, less B; cool light = more B, less R)
    if rng.random() > 0.5:
        scale_R = rng.uniform(1.1, 1.4)        # warm cast
        scale_B = rng.uniform(0.7, 0.9)
    else:
        scale_R = rng.uniform(0.7, 0.9)        # cool cast
        scale_B = rng.uniform(1.1, 1.4)

    scales = [scale_B, scale_G, scale_R]
    out    = img.astype(np.float32).copy()
    for c, s in enumerate(scales):
        out[:, :, c] *= s
    out = _clip(out)
    return out, {"scale_B": round(scale_B, 3),
                "scale_G": round(scale_G, 3),
                "scale_R": round(scale_R, 3)}


def main():
    test_dir = Path("../data/metal_nut/test").resolve()
    n_samples = 4
    seed = 41
    random_pick = True  # False = first N images

    rng = random.Random(seed)

    all_paths = collect_test_images(test_dir)
   
    samples = choose_samples(all_paths, n_samples, rng, random_pick=random_pick)
    print(f"Found {len(all_paths)} total images. Showing {len(samples)} samples.")

    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    if len(samples) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, img_path in enumerate(samples):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        aug_bgr, params = apply_white_balance(img_bgr, rng)
        cls_name = img_path.parent.name

        axes[0, i].imshow(_bgr_to_rgb(img_bgr))
        axes[0, i].set_title(f"Original | {cls_name}\n{img_path.name}")
        axes[0, i].axis("off")

        axes[1, i].imshow(_bgr_to_rgb(aug_bgr))
        axes[1, i].set_title(
            f"White Balance Shifted | {cls_name}\n"
            f"B={params['scale_B']} G={params['scale_G']} R={params['scale_R']}"
        )
        axes[1, i].axis("off")

        print(
            f"[{cls_name}] {img_path.name} -> "
            f"scale_B={params['scale_B']}, scale_G={params['scale_G']}, scale_R={params['scale_R']}"
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()