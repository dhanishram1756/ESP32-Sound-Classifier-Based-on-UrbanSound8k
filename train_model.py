"""Train a compact float32 UrbanSound8K classifier and export a TFLite model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sound_features import (
    FEATURE_BINS,
    FEATURE_COUNT,
    FRAME_COUNT,
    FRAME_SAMPLES,
    SAMPLE_RATE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/urbansound8k_features.npz"),
        help="Output from extract_features.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for sound_classifier.tflite and metadata.json",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    with np.load(args.features, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int32)
        class_names = data["class_names"].astype(str).tolist()
        saved_rate = int(data["sample_rate"])

    if saved_rate != SAMPLE_RATE or features.ndim != 2:
        raise ValueError("Feature dataset has an incompatible sample rate or shape")
    if features.shape[1] != FEATURE_COUNT or labels.size != features.shape[0]:
        raise ValueError("Feature and label dimensions are inconsistent")
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature dataset contains non-finite values")

    train_mask = (folds >= 1) & (folds <= 8)
    validation_mask = folds == 9
    test_mask = folds == 10
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Dataset must contain examples from folds 1 through 10")

    train_features = features[train_mask]
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    normalized = (features - mean) / scale
    num_classes = len(class_names)
    if set(np.unique(labels)) != set(range(num_classes)):
        raise ValueError("Labels do not match the class_names array")

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_COUNT,), name="features"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="scores"),
        ],
        name="urbansound8k_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        normalized[train_mask],
        labels[train_mask],
        validation_data=(normalized[validation_mask], labels[validation_mask]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    test_probabilities = model.predict(
        normalized[test_mask], batch_size=args.batch_size, verbose=0
    )
    test_predictions = test_probabilities.argmax(axis=1)
    test_labels = labels[test_mask]
    confusion = np.bincount(
        test_labels * num_classes + test_predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    f1_denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    class_f1 = np.divide(
        2.0 * confusion.diagonal(),
        f1_denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=f1_denominator != 0,
    )
    test_accuracy = float(np.mean(test_predictions == test_labels))
    test_macro_f1 = float(class_f1.mean())
    print(
        f"Fold 10 test accuracy: {test_accuracy:.4f}; "
        f"macro F1: {test_macro_f1:.4f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "sound_classifier.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    model_path.write_bytes(converter.convert())

    metadata = {
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "frame_count": FRAME_COUNT,
        "feature_bins": list(FEATURE_BINS),
        "feature_count": FEATURE_COUNT,
        "class_names": class_names,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "test_fold": 10,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "model_hidden_units": [64, 32],
        "dropout_rate": 0.15,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved TFLite model to {model_path}")
    print(f"Saved model metadata to {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
