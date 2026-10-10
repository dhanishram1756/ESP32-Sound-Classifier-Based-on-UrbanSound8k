"""Convert a float32 TFLite model and its metadata into an Arduino header."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sound_features import FEATURE_COUNT


def c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def float_array(name: str, values: list[float]) -> str:
    formatted = ", ".join(f"{value:.9g}f" for value in values)
    return f"const float {name}[{len(values)}] = {{{formatted}}};"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/sound_classifier.tflite"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/metadata.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("firmware/esp32_sound_classifier/model_data.h"),
    )
    args = parser.parse_args()

    model = args.model.read_bytes()
    if not model:
        raise ValueError(f"TFLite model is empty: {args.model}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    means = metadata["feature_mean"]
    scales = metadata["feature_scale"]
    labels = metadata["class_names"]
    if (
        metadata.get("feature_count") != FEATURE_COUNT
        or len(means) != FEATURE_COUNT
        or len(scales) != FEATURE_COUNT
        or not labels
    ):
        raise ValueError("Model metadata does not match the expected feature layout")
    if any(not np.isfinite(value) for value in means + scales):
        raise ValueError("Model metadata contains non-finite normalization values")

    model_bytes = ",\n".join(
        "    " + ", ".join(f"0x{byte:02x}" for byte in model[start : start + 12])
        for start in range(0, len(model), 12)
    )
    label_array = ", ".join(c_string(label) for label in labels)
    contents = f"""#pragma once

#include <cstdint>

alignas(8) const unsigned char g_sound_model[] = {{
{model_bytes}
}};
const unsigned int g_sound_model_len = sizeof(g_sound_model);
constexpr unsigned int kFeatureCount = {FEATURE_COUNT};
constexpr unsigned int kClassCount = {len(labels)};
{float_array("g_feature_mean", means)}
{float_array("g_feature_scale", scales)}
const char* const g_class_names[kClassCount] = {{{label_array}}};
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(contents, encoding="ascii")
    print(f"Wrote {len(model)} model bytes and {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    main()
