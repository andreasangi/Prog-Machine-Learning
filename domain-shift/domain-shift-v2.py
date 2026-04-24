import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_FOLDERS = ["bent", "color", "flip", "good", "scratch"]
VALID_EXTS = {".png", ".jpg", ".jpeg"}


def _clip(img: np.ndarray) -> np.ndarray:
    """Clip float image to [0, 255] and cast to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def apply_exposure(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Simulate exposure variation via linear scaling.
    NEW_PIXEL = alpha * OLD_PIXEL + beta

    alpha < 1  → under-exposure (e.g. insufficient light, fast shutter)
    alpha > 1  → over-exposure (e.g. blown highlights, long exposure)
    beta       → additive offset (dark current / ambient offset)

    Industrial case: line lighting intensity varies with voltage
    fluctuations, lamp aging, and controller settings.
    Plausible range: alpha ∈ [0.5, 1.7], beta ∈ [-30, 30]
    """
    alpha = rng.uniform(0.5, 1.7)
    beta  = rng.uniform(-30, 30)
    out   = _clip(img.astype(np.float32) * alpha + beta)
    return out, {"alpha": round(alpha, 3), "beta": round(beta, 3)}

def apply_gamma(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Gamma correction to simulate different sensor response curves.

    gamma < 1  → image brightened (as if sensor is more sensitive)      -- lifts shadows/midtones
    gamma > 1  → image darkened         -- compresses shadows/midtones

    Industrial case: different camera models (or firmware versions)
    apply different gamma tables in-sensor. 
    Mild miscalibration between camera units (around 1 gamma) or 
    gamma correction accidentally enabled/disabled (gamma  0.45 or 2.2).
    Plausible range for fluctations: gamma ∈ [0.45, 2.2]
    """
    gamma     = rng.uniform(0.45, 2.2)
    lut       = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    out       = cv2.LUT(img, lut)
    return out, {"gamma": round(gamma, 3)}
