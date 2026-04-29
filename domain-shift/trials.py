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

def apply_perspective(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Mild perspective warp to simulate camera tilt or off-axis mounting.

    Four corner points are perturbed independently by a small random
    amount, then cv2.getPerspectiveTransform maps them to the full frame.

    Industrial case:
    slight camera tilt after maintenace, fixture wobble, imperfect mounting,
    parts not planar to the sensor, etc. 

    Plausible range: corner jitter ∈ [0, perturb_frac × min(h,w)]
                     perturb_frac ∈ [0.02, 0.07]  (2–7% of image size)
    """
    h, w   = img.shape[:2]
    frac   = rng.uniform(0.02, 0.07)
    jitter = frac * min(h, w)   # converts to px, and ensures perturbation scales with image size

    def j():
        return rng.uniform(-jitter, jitter)

    src = np.float32([[0,   0  ],
                      [w-1, 0  ],
                      [w-1, h-1],
                      [0,   h-1]])
    dst = np.float32([[0   + j(), 0   + j()],
                      [w-1 + j(), 0   + j()],
                      [w-1 + j(), h-1 + j()],
                      [0   + j(), h-1 + j()]])

    M   = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return out, {"perturb_frac": round(frac, 4),
                 "dst_corners":  dst.tolist()}



def main():
    test_dir = Path("../data/metal_nut/test").resolve()
    n_samples = 4
    seed = 47
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

        aug_bgr, params = apply_perspective(img_bgr, rng)
        cls_name = img_path.parent.name

        axes[0, i].imshow(_bgr_to_rgb(img_bgr))
        axes[0, i].set_title(f"Original | {cls_name}\n{img_path.name}")
        axes[0, i].axis("off")

        axes[1, i].imshow(_bgr_to_rgb(aug_bgr))
        subtitle = f"perturb_frac={params['perturb_frac']}"

        axes[1, i].set_title(
            f"Perspective Shifted | {cls_name}\n{subtitle}"
        )
        axes[1, i].axis("off")

        print(
            f"[{cls_name}] {img_path.name} -> "
            f"perturb_frac={params['perturb_frac']}"
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()