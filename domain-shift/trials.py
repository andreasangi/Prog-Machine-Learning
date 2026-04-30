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
    Perspective warp derived from physical camera tilt angles.

    Builds the homography from explicit pitch and roll angles, this guarantees that circles
    remain ellipses, straight lines remain straight, and the warp corresponds
    to a real camera position.

    Physical model: pinhole camera, planar object, orthographic approximation
    for small angles. Focal length estimated from image size.

    Industrial case: camera remounted after maintenance with slight tilt,
    fixture not perfectly level, imperfect mounting, Scheimpflug tilt for depth of field control.

    Plausible range:
        pitch (x-tilt): ±15°  — camera nodding forward/backward
        roll  (y-tilt): ±15°  — camera tilting left/right
        Both angles independently sampled.
    """
    h, w = img.shape[:2]

    # Estimated focal length: reasonable assumption for machine vision macro lens
    # Approximately equal to image width for a ~50° horizontal FOV
    f = w

    pitch_deg = rng.uniform(-15, 15)   # tilt around x axis (forward/back)
    roll_deg  = rng.uniform(-15, 15)   # tilt around y axis (left/right)

    pitch = np.deg2rad(pitch_deg)
    roll  = np.deg2rad(roll_deg)

    # Rotation matrices around x and y axes
    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(pitch), -np.sin(pitch)],
                   [0,  np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)

    Ry = np.array([[ np.cos(roll), 0, np.sin(roll)],
                   [0,             1,           0  ],
                   [-np.sin(roll), 0, np.cos(roll)]], dtype=np.float32)

    R = Ry @ Rx   # combined rotation

    # Camera intrinsic matrix, principal point at image centre
    cx, cy = w / 2, h / 2
    K = np.array([[f,  0, cx],
                  [0,  f, cy],
                  [0,  0,  1]], dtype=np.float32)

    # Homography: H = K @ R @ K_inv
    # Valid for planar scene viewed from different angles
    H = K @ R @ np.linalg.inv(K)
    H = H / H[2, 2]   # normalise so H[2,2] = 1

    out = cv2.warpPerspective(img, H, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    return out, {
        "pitch_deg": round(pitch_deg, 2),
        "roll_deg":  round(roll_deg,  2),
    }


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
        subtitle = (
            f"pitch={params['pitch_deg']} roll={params['roll_deg']}"
        )

        axes[1, i].set_title(
            f"Perspective Shifted | {cls_name}\n{subtitle}"
        )
        axes[1, i].axis("off")

        print(
            f"[{cls_name}] {img_path.name} -> "
            f"pitch_deg={params['pitch_deg']}, roll_deg={params['roll_deg']}"
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()