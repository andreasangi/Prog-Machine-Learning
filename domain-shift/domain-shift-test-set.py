import random
import json
from pathlib import Path

import cv2

from domain-shift-fucntions import (
    apply_exposure, apply_gamma, apply_white_balance, apply_noise,
    apply_jpeg, apply_blur, apply_vignette, apply_shadow, apply_contrast,
    apply_affine, apply_perspective, apply_specular
)

CLASS_FOLDERS = ["bent", "color", "flip", "good", "scratch"]
VALID_EXTS = {".png", ".jpg", ".jpeg"}

ROOT = Path("../data/metal_nut/test").resolve()
OUT  = Path("./augmented_subset").resolve()

# Relative to this script's location (projectRoot/domain-shift/)
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data" / "metal_nut" / "test"
OUTPUT_ROOT = SCRIPT_DIR / "augmented_test"
