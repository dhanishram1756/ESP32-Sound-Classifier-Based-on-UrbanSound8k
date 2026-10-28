"""Train a compact float32 UrbanSound8K classifier and export a TFLite model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sound_features import (
    FEATURE_BINS,
    FEATURE_COUNT,
    FRAME_COUNT,
    FRAME_SAMPLES,
    SAMPLE_RATE,
)


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Save loss and accuracy curves for the training run."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["loss"], label="Training", color="#1677b9")
    axes[0].plot(epochs, history["val_loss"], label="Validation", color="#ef8354")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Sparse categorical cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["accuracy"], label="Training", color="#1677b9")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation", color="#ef8354")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(
    confusion: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    """Save raw counts and normalized percentages side by side."""
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals != 0,
    )

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, values, title, fmt in (
        (axes[0], confusion, "Confusion matrix (counts)", "d"),
        (axes[1], normalized, "Confusion matrix (row-normalized)", ".1%"),
    ):
        image = axis.imshow(values, interpolation="nearest", cmap="Blues")
        axis.set(
            title=title,
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted class",
            ylabel="True class",
        )
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for row in range(len(class_names)):
            for column in range(len(class_names)):
                text = f"{values[row, column]:{fmt}}" if fmt == "d" else f"{values[row, column]:{fmt}}"
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if values[row, column] > values.max() / 2 else "black",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    with np.load(args.features, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int32)
        filenames = data["filenames"].astype(str)
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
            tf.keras.layers.GaussianNoise(0.08),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="scores"),
        ],
        name="urbansound8k_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        normalized[train_mask],
        labels[train_mask],
        validation_data=(normalized[validation_mask], labels[validation_mask]),
        shuffle=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, min_delta=1e-4, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
        ],
        verbose=2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history.history, args.output_dir / "training_curves.png")

    test_probabilities = model.predict(
        normalized[test_mask], batch_size=args.batch_size, verbose=0
    )
    test_predictions = test_probabilities.argmax(axis=1)
    test_labels = labels[test_mask]
    # Window-level confusion is displayed below, but clip-level metrics better
    # represent whether a complete UrbanSound8K recording is classified correctly.
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
    window_accuracy = float(np.mean(test_predictions == test_labels))

    clip_probabilities: dict[str, list[np.ndarray]] = {}
    clip_labels: dict[str, int] = {}
    test_filenames = filenames[test_mask]
    for filename, probabilities, label in zip(
        test_filenames, test_probabilities, test_labels
    ):
        clip_probabilities.setdefault(str(filename), []).append(probabilities)
        clip_labels[str(filename)] = int(label)

    clip_predictions = []
    clip_true_labels = []
    for filename, window_scores in clip_probabilities.items():
        clip_predictions.append(int(np.mean(window_scores, axis=0).argmax()))
        clip_true_labels.append(clip_labels[filename])
    clip_predictions_array = np.asarray(clip_predictions)
    clip_true_labels_array = np.asarray(clip_true_labels)
    test_accuracy = float(np.mean(clip_predictions_array == clip_true_labels_array))
    clip_confusion = np.bincount(
        clip_true_labels_array * num_classes + clip_predictions_array,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    clip_f1_denominator = (
        clip_confusion.sum(axis=0) + clip_confusion.sum(axis=1)
    )
    clip_class_f1 = np.divide(
        2.0 * clip_confusion.diagonal(),
        clip_f1_denominator,
        out=np.zeros(num_classes, dtype=np.float64),
        where=clip_f1_denominator != 0,
    )
    test_macro_f1 = float(clip_class_f1.mean())
    confusion = clip_confusion
    plot_confusion_matrix(confusion, class_names, args.output_dir / "confusion_matrix.png")
    np.savetxt(
        args.output_dir / "confusion_matrix.csv",
        confusion,
        fmt="%d",
        delimiter=",",
        header=",".join(class_names),
        comments="",
    )
    print(
        f"Fold 10 window accuracy: {window_accuracy:.4f}; "
        f"clip accuracy: {test_accuracy:.4f}; macro F1: {test_macro_f1:.4f}"
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
        "model_hidden_units": [256, 128],
        "dropout_rates": [0.25, 0.20],
        "evaluation_unit": "clip",
        "training_epochs_requested": args.epochs,
        "epochs_completed": len(history.history["loss"]),
        "training_curves": "training_curves.png",
        "confusion_matrix": "confusion_matrix.png",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved TFLite model to {model_path}")
    print(f"Saved model metadata to {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
