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

def apply_blur(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Motion blur (directional) OR defocus blur (isotropic), chosen randomly.

    Motion blur:   linear kernel at a random angle -
                   models conveyor, camera vibration or object not perfectly still during exposure.
    Defocus blur:  Gaussian kernel
                   Object not on focal plane,
                   models depth-of-field variation when objects have different 
                   heights on the inspection tapis.

    Plausible kernel sizes: motion length ∈ [5, 25 px], angle ∈ [0°, 180°]
                            defocus sigma ∈ [1.5, 6.0]
    """
    kind = rng.choice(["motion", "defocus"])
    params: dict = {"kind": kind}

    if kind == "motion":
        length = rng.randint(5, 25)
        angle  = rng.uniform(0, 180)
        params.update({"length": length, "angle_deg": round(angle, 1)})

        # Build a line at the desired angle to create the kernel
        kernel = np.zeros((length, length), dtype=np.float32)
        cx, cy = length // 2, length // 2
        angle_rad = np.deg2rad(angle)
        x1 = int(cx - (length // 2) * np.cos(angle_rad))
        y1 = int(cy - (length // 2) * np.sin(angle_rad))
        x2 = int(cx + (length // 2) * np.cos(angle_rad))
        y2 = int(cy + (length // 2) * np.sin(angle_rad))
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)

        kernel  = kernel / kernel.sum()             # re-normalise after rotation
            # warpAffine uses bilinear interpolation when rotating, which distributes energy 
            # across neighbouring pixels and can reduce the kernel sum below 1, causing 
            # the blurred image to darken slightly

        out     = cv2.filter2D(img, -1, kernel)

    else:  # defocus
        sigma = rng.uniform(1.5, 6.0)
        ksize = int(sigma * 6) | 1                 # must be odd
        params.update({"sigma": round(sigma, 2), "ksize": ksize})
        out   = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    return out, params



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

        aug_bgr, params = apply_blur(img_bgr, rng)
        cls_name = img_path.parent.name

        axes[0, i].imshow(_bgr_to_rgb(img_bgr))
        axes[0, i].set_title(f"Original | {cls_name}\n{img_path.name}")
        axes[0, i].axis("off")

        axes[1, i].imshow(_bgr_to_rgb(aug_bgr))
        if params["kind"] == "motion":
            subtitle = f"motion | len={params['length']} angle={params['angle_deg']}°"
        else:
            subtitle = f"defocus | sigma={params['sigma']} k={params['ksize']}"

        axes[1, i].set_title(
            f"Blur Shifted | {cls_name}\n{subtitle}"
        )
        axes[1, i].axis("off")

        if params["kind"] == "motion":
            print(
                f"[{cls_name}] {img_path.name} -> "
                f"kind=motion, length={params['length']}, angle_deg={params['angle_deg']}"
            )
        else:
            print(
                f"[{cls_name}] {img_path.name} -> "
                f"kind=defocus, sigma={params['sigma']}, ksize={params['ksize']}"
            )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()