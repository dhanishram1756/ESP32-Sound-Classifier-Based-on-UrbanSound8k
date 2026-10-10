"""Feature extraction shared by the UrbanSound8K preparation script."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 500
FRAME_COUNT = 32
HOP_SAMPLES = 8_000
FEATURE_BINS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
FEATURE_COUNT = FRAME_COUNT * len(FEATURE_BINS)
_WINDOW = (
    0.5
    - 0.5
    * np.cos(
        2.0 * np.pi * np.arange(FRAME_SAMPLES, dtype=np.float64)
        / (FRAME_SAMPLES - 1)
    )
)
_WINDOW_ENERGY = float(np.dot(_WINDOW, _WINDOW))
_FEATURE_BINS_ARRAY = np.asarray(FEATURE_BINS, dtype=np.intp)


def extract_features(samples: np.ndarray) -> np.ndarray:
    """Return 384 log-power Goertzel features for one second of audio."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    required_samples = FRAME_COUNT * FRAME_SAMPLES
    if audio.size != required_samples:
        raise ValueError(f"Expected {required_samples} samples, got {audio.size}")

    frames = audio.astype(np.float64).reshape(FRAME_COUNT, FRAME_SAMPLES)
    frames -= frames.mean(axis=1, keepdims=True)
    frames *= _WINDOW

    spectrum = np.fft.rfft(frames, axis=1)
    power = np.square(np.abs(spectrum[:, _FEATURE_BINS_ARRAY]))
    log_power = np.log10(np.maximum(power / _WINDOW_ENERGY, 1e-12))
    return log_power.astype(np.float32).reshape(-1)
