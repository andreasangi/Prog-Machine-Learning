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

def apply_noise(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Gaussian read-noise + optional salt-and-pepper dead/hot pixels.

    Gaussian sigma models thermal / read noise (low-light or high-gain).
    Salt-and-pepper fraction models sensor defects or transmission errors.

    Industrial justification: industrial cameras operating at high gain
    (low-light) show significant read noise; older sensors develop dead pixels.
    Plausible range: sigma ∈ [5, 40], sp_fraction ∈ [0, 0.005]
    """
    sigma      = rng.uniform(5, 40)
    sp_frac    = rng.uniform(0.0, 0.005)

    # Gaussian
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out   = _clip(img.astype(np.float32) + noise)

    # Salt-and-pepper
    n_pixels = int(sp_frac * img.shape[0] * img.shape[1])
    if n_pixels > 0:
        # Salt (white)
        ys = np.random.randint(0, img.shape[0], n_pixels // 2)
        xs = np.random.randint(0, img.shape[1], n_pixels // 2)
        out[ys, xs] = 255
        # Pepper (black)
        ys = np.random.randint(0, img.shape[0], n_pixels // 2)
        xs = np.random.randint(0, img.shape[1], n_pixels // 2)
        out[ys, xs] = 0

    return out, {"gaussian_sigma": round(sigma, 2),
                 "sp_fraction":    round(sp_frac, 5)}


def main():
    test_dir = Path("../data/metal_nut/test").resolve()
    n_samples = 4
    seed = 41
    random_pick = True  # False = first N images

    rng = random.Random(seed)
    np.random.seed(seed)

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

        aug_bgr, params = apply_noise(img_bgr, rng)
        cls_name = img_path.parent.name

        axes[0, i].imshow(_bgr_to_rgb(img_bgr))
        axes[0, i].set_title(f"Original | {cls_name}\n{img_path.name}")
        axes[0, i].axis("off")

        axes[1, i].imshow(_bgr_to_rgb(aug_bgr))
        axes[1, i].set_title(
            f"Noise Shifted | {cls_name}\n"
            f"sigma={params['gaussian_sigma']} sp={params['sp_fraction']}"
        )
        axes[1, i].axis("off")

        print(
            f"[{cls_name}] {img_path.name} -> "
            f"gaussian_sigma={params['gaussian_sigma']}, sp_fraction={params['sp_fraction']}"
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()