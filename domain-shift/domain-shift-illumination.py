"""
Simple baseline domain-shift visualizer (illumination only).

Goal for this first version:
- Read random images from ../data/metal_nut/test/{bent,color,flip,good,scratch}
- Apply a lightweight illumination shift
- Show original vs shifted with matplotlib

No file saving yet.
"""

from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = ["bent", "color", "flip", "good", "scratch"]
IMG_EXTS = {".png", ".jpg"}


def collect_images(test_root: Path):
	"""Collect all image paths grouped by class name."""
	images_by_class = {}
	for cls in CLASS_NAMES:
		cls_dir = test_root / cls
		if not cls_dir.exists():
			images_by_class[cls] = []
			continue

		paths = [
			p for p in sorted(cls_dir.iterdir())
			if p.is_file() and p.suffix.lower() in IMG_EXTS
		]
		images_by_class[cls] = paths
	return images_by_class


def apply_baseline_illumination_shift(img_bgr: np.ndarray):
	"""
	Baseline shift = exposure + brightness/contrast + white-balance drift.
	Returns shifted image and a dict with applied params.
	"""
	img = img_bgr.astype(np.float32)

	# 1) Exposure-like scaling
	exposure_scale = random.uniform(0.75, 1.30)
	img = img * exposure_scale

	# 2) Brightness/contrast
	alpha_contrast = random.uniform(0.90, 1.15)   # contrast
	beta_brightness = random.uniform(-18, 18)     # brightness offset
	img = img * alpha_contrast + beta_brightness

	# 3) White balance drift (per-channel multipliers in BGR order)
	wb_b = random.uniform(0.92, 1.08)
	wb_g = random.uniform(0.95, 1.05)
	wb_r = random.uniform(0.92, 1.08)
	img[..., 0] *= wb_b
	img[..., 1] *= wb_g
	img[..., 2] *= wb_r

	img = np.clip(img, 0, 255).astype(np.uint8)
	params = {
		"exposure": exposure_scale,
		"contrast": alpha_contrast,
		"brightness": beta_brightness,
		"wb_b": wb_b,
		"wb_g": wb_g,
		"wb_r": wb_r,
	}
	return img, params


def bgr_to_rgb(img_bgr: np.ndarray):
	return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def show_samples(images_by_class, samples_per_class=1, seed=43):
	"""Plot original vs shifted samples."""
	random.seed(seed)
	np.random.seed(seed)

	selected = []
	for cls in CLASS_NAMES:
		candidates = images_by_class.get(cls, [])
		if len(candidates) == 0:
			continue
		k = min(samples_per_class, len(candidates))
		selected.extend((cls, p) for p in random.sample(candidates, k))

	if len(selected) == 0:
		print("No images found. Check dataset path.")
		return

	n_rows = len(selected)
	fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows))

	if n_rows == 1:
		axes = np.array([axes])

	for i, (cls, img_path) in enumerate(selected):
		img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
		if img_bgr is None:
			print(f"Skipping unreadable image: {img_path}")
			continue

		shifted_bgr, params = apply_baseline_illumination_shift(img_bgr)

		axes[i, 0].imshow(bgr_to_rgb(img_bgr))
		axes[i, 0].set_title(f"Original | {cls}\n{img_path.name}")
		axes[i, 0].axis("off")

		param_txt = (
			f"exp={params['exposure']:.2f}, c={params['contrast']:.2f}, "
			f"b={params['brightness']:.1f}\n"
			f"wb(B,G,R)=({params['wb_b']:.2f},{params['wb_g']:.2f},{params['wb_r']:.2f})"
		)
		axes[i, 1].imshow(bgr_to_rgb(shifted_bgr))
		axes[i, 1].set_title(f"Shifted | {cls}\n{param_txt}")
		axes[i, 1].axis("off")

		print(f"[{cls}] {img_path.name} -> {param_txt}")

	plt.tight_layout()
	plt.show()


def main():
	# Resolve dataset path relative to this script location
	script_dir = Path(__file__).resolve().parent
	test_root = (script_dir / "../data/metal_nut/test").resolve()

	print(f"Using test path: {test_root}")
	if not test_root.exists():
		print("Test path does not exist. Update the path in main().")
		return

	images_by_class = collect_images(test_root)
	for cls in CLASS_NAMES:
		print(f"{cls:8s}: {len(images_by_class.get(cls, []))} images")

	# First simple preview: 1 random sample per class (when available)
	show_samples(images_by_class, samples_per_class=1, seed=43)


if __name__ == "__main__":
	main()