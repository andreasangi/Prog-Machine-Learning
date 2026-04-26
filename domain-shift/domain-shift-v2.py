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

def apply_white_balance(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Per-channel multiplicative drift to simulate white-balance miscalibration.

    R and B are anti-correlated to model colour temperature shift along the
    warm/cool axis (warm = more R, less B; cool = more B, less R).
    G stays nearly stable as cameras are designed around the green channel.

    Industrial case: switching between fluorescent, LED, halogen and sodium-
    vapour lighting changes the illuminant spectrum; a fixed white-balance
    preset introduces a colour cast.
    Plausible range: per-channel scale ∈ [0.75, 1.25]
    """
    warm = rng.random() > 0.5          # True → warm cast, False → cool cast

    scale_G = rng.uniform(0.90, 1.10)
    scale_R = rng.uniform(1.10, 1.40) if warm else rng.uniform(0.70, 0.90)
    scale_B = rng.uniform(0.70, 0.90) if warm else rng.uniform(1.10, 1.40)

    out = img.astype(np.float32).copy()
    for c, s in enumerate([scale_B, scale_G, scale_R]):
        out[:, :, c] *= s
    out = _clip(out)
    return out, {"scale_B": round(scale_B, 3),
                 "scale_G": round(scale_G, 3),
                 "scale_R": round(scale_R, 3),
                 "cast":    "warm" if warm else "cool"}

def apply_noise(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Gaussian read-noise + optional salt-and-pepper dead/hot pixels.

    Gaussian sigma models thermal (read) noise (low-light or high-gain).
    Salt-and-pepper fraction models sensor defects, hot/dead pixels.

    Industrial case: industrial cameras operating at high gain
    (low-light) show significant read noise, and older sensors develop dead pixels.
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
