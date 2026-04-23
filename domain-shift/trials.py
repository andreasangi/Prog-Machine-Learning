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

def apply_gamma(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Gamma correction to simulate different sensor response curves.

    gamma < 1  → image brightened (as if sensor is more sensitive)      -- lifts shadows/midtones
    gamma > 1  → image darkened         -- compresses shadows/midtones

    Industrial case: different camera models (or firmware versions)
    apply different gamma tables in-sensor. Plausible range: gamma ∈ [0.45, 2.2]
    """
    gamma     = rng.uniform(0.45, 2.2)
    lut       = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    out       = cv2.LUT(img, lut)
    return out, {"gamma": round(gamma, 3)}


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

        aug_bgr, params = apply_gamma(img_bgr, rng)
        cls_name = img_path.parent.name

        axes[0, i].imshow(_bgr_to_rgb(img_bgr))
        axes[0, i].set_title(f"Original | {cls_name}\n{img_path.name}")
        axes[0, i].axis("off")

        axes[1, i].imshow(_bgr_to_rgb(aug_bgr))
        axes[1, i].set_title(
            f"Gamma Shifted | {cls_name}\ngamma={params['gamma']}"
        )
        axes[1, i].axis("off")

        print(f"[{cls_name}] {img_path.name} -> gamma={params['gamma']}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()