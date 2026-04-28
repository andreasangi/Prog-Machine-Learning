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

def apply_jpeg(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Simulate JPEG compression artifacts by encoding and re-decoding.

    Lower quality → stronger block artifacts (8×8 DCT blocks visible).

    Industrial case: images transmitted over network links (GigE
    Vision with software compression, or IP cameras) are often JPEG-encoded.
    Plausible range: quality ∈ [20, 60]
    """
    quality = rng.randint(20, 60)
    _, buf  = cv2.imencode('.jpg', img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    out     = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out, {"jpeg_quality": quality}

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

def apply_vignette(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Apply a smooth radial brightness falloff (vignetting) that darkens toward the image corners.

    Implemented as a 2D Gaussian mask centred at (cx, cy) with spread sigma.
    Strength parameter controls how dark the corners get.

    Industrial case: all lenses exhibit natural vignetting;
    many real camera+lens systems exhibit relative-illumination roll-off toward the
    periphery due to optical and mechanical vignetting; severity depends on 
    lens design, aperture, and sensor/lens matching.
    Plausible range: strength ∈ [0.3, 0.8], sigma_frac ∈ [0.5, 0.9]
    """
    h, w   = img.shape[:2]
    strength    = rng.uniform(0.3, 0.8)   # how dark corners get (0 = no effect)
    sigma_frac  = rng.uniform(0.5, 0.9)   # Gaussian spread as fraction of image size

    sigma_x = w * sigma_frac
    sigma_y = h * sigma_frac
    cx, cy  = w / 2, h / 2

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # 2D Gaussian mask with centre=1 and corners approaching (1 - strength)
    mask = np.exp(-((X - cx) ** 2 / (2 * sigma_x ** 2) +
                    (Y - cy) ** 2 / (2 * sigma_y ** 2)))

    # Scale mask so centre = 1 and corners = (1 - strength)
    mask = 1.0 - strength * (1.0 - mask)
    mask = mask[:, :, np.newaxis]            # broadcast over channels

    out = _clip(img.astype(np.float32) * mask)
    return out, {"strength":   round(strength, 3),
                 "sigma_frac": round(sigma_frac, 3)}


def apply_shadow(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """
    Directional gradient shadow: a linear brightness ramp across the image.

    Models uneven scene illumination from a single off-axis light source. 

    Industrial case: lamp repositioning or replacement, single-sided ring light failure,
                    conveyor edge shadow, factory window light...

    direction: 'horizontal' | 'vertical' | 'diagonal'
    dark_side:  which edge is darkened (0 = left/top, 1 = right/bottom)
    intensity:  fraction of brightness lost at the dark edge [0.2, 0.6]
    """
    direction  = rng.choice(["horizontal", "vertical", "diagonal"])
    dark_side  = rng.randint(0, 1)
    intensity  = rng.uniform(0.2, 0.6)

    h, w = img.shape[:2]

    if direction == "horizontal":
        ramp = np.linspace(0, 1, w, dtype=np.float32)
        mask = np.tile(ramp, (h, 1))
    elif direction == "vertical":
        ramp = np.linspace(0, 1, h, dtype=np.float32)
        mask = np.tile(ramp[:, np.newaxis], (1, w))
    else:  
        ramp_x = np.linspace(0, 1, w, dtype=np.float32)
        ramp_y = np.linspace(0, 1, h, dtype=np.float32)
        mask   = np.outer(ramp_y, ramp_x)

    if dark_side == 0:
        mask = 1.0 - mask

    # bright side = no change, dark side = multiply by (1 - intensity)
    scale = 1.0 - intensity * (1.0 - mask)
    scale = scale[:, :, np.newaxis]

    out = _clip(img.astype(np.float32) * scale)
    return out, {"direction": direction, "dark_side": dark_side,
                 "intensity": round(intensity, 3)}


