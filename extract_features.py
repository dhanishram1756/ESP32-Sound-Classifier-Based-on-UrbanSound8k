"""Extract fixed-size audio windows and Goertzel features from UrbanSound8K."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from sound_features import (
    FRAME_COUNT,
    FRAME_SAMPLES,
    HOP_SAMPLES,
    SAMPLE_RATE,
    extract_features,
)


def load_audio(path: Path) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, dtype=np.float64)

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Audio contains non-finite values: {path}")

    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(int(source_rate), SAMPLE_RATE)
        audio = resample_poly(
            audio, SAMPLE_RATE // divisor, int(source_rate) // divisor
        )

    return audio.astype(np.float32)


def window_starts(audio_length: int) -> list[int]:
    window_length = FRAME_COUNT * FRAME_SAMPLES
    if audio_length <= window_length:
        return [0]
    starts = list(range(0, audio_length - window_length + 1, HOP_SAMPLES))
    final_start = audio_length - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="UrbanSound8K root containing audio/ and metadata/UrbanSound8K.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/urbansound8k_features.npz"),
        help="Compressed feature dataset path",
    )
    args = parser.parse_args()

    metadata_path = args.dataset / "metadata" / "UrbanSound8K.csv"
    audio_root = args.dataset / "audio"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"UrbanSound8K metadata not found: {metadata_path}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"UrbanSound8K audio directory not found: {audio_root}")

    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    if not rows:
        raise ValueError(f"No dataset rows found in {metadata_path}")

    class_names: dict[int, str] = {}
    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    fold_rows: list[int] = []
    file_rows: list[str] = []
    started = time.monotonic()

    for row_index, row in enumerate(rows, start=1):
        class_id = int(row["classID"])
        class_name = row["class"]
        if class_id in class_names and class_names[class_id] != class_name:
            raise ValueError(f"Conflicting names found for class ID {class_id}")
        class_names[class_id] = class_name
        fold = int(row["fold"])
        filename = row["slice_file_name"]
        audio_path = audio_root / f"fold{fold}" / filename
        audio = load_audio(audio_path)
        window_length = FRAME_COUNT * FRAME_SAMPLES

        for start in window_starts(audio.size):
            window = audio[start : start + window_length]
            if window.size < window_length:
                window = np.pad(window, (0, window_length - window.size))
            feature_rows.append(extract_features(window))
            label_rows.append(class_id)
            fold_rows.append(fold)
            file_rows.append(filename)

        if row_index % 100 == 0 or row_index == len(rows):
            elapsed = time.monotonic() - started
            print(
                f"Processed {row_index}/{len(rows)} clips; "
                f"{len(feature_rows)} windows; {elapsed:.1f}s elapsed",
                flush=True,
            )

    class_ids = sorted(class_names)
    if class_ids != list(range(len(class_ids))):
        raise ValueError(f"Expected contiguous class IDs starting at zero, got {class_ids}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(feature_rows).astype(np.float32),
        labels=np.asarray(label_rows, dtype=np.int64),
        folds=np.asarray(fold_rows, dtype=np.uint8),
        filenames=np.asarray(file_rows, dtype=np.str_),
        class_names=np.asarray([class_names[i] for i in class_ids], dtype=np.str_),
        sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int32),
    )
    print(
        f"Saved {len(feature_rows)} windows from {len(rows)} clips "
        f"({len(class_ids)} classes) to {args.output}"
    )


if __name__ == "__main__":
    main()
