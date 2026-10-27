"""Optional model explainability, evaluation and deployment export utilities."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def gradcam_heatmap(image_bytes: bytes, model_path: str, labels_path: str | None = None,
                    class_index: int | None = None, image_size: int | None = None) -> dict[str, Any]:
    """Create Grad-CAM for compatible single-input CNNs with a Conv2D layer.

    Grad-CAM is an approximate visual explanation, not proof of model reasoning.
    The model is expected to use the MobileNetV2 preprocessing used by this project.
    """
    import io
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    model = tf.keras.models.load_model(path)
    if isinstance(model.input_shape, list) or len(model.input_shape) != 4:
        raise ValueError("Grad-CAM currently supports single-input image models only.")
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        # Handle a common nested backbone layout.
        for outer in reversed(model.layers):
            if isinstance(outer, tf.keras.Model):
                conv_layers = [layer for layer in outer.layers if isinstance(layer, tf.keras.layers.Conv2D)]
                if conv_layers:
                    break
    if not conv_layers:
        raise ValueError("No Conv2D layer found; Grad-CAM needs a convolutional model.")
    conv_layer = conv_layers[-1]
    try:
        grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    except Exception as exc:
        raise ValueError("Could not connect the final convolution layer to model output. This architecture may not support Grad-CAM.") from exc
    shape = model.input_shape
    h = int(image_size or shape[1] or 224)
    w = int(image_size or shape[2] or 224)
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB").resize((w, h))
        arr = np.asarray(rgb, dtype=np.float32)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr.copy(), 0))
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(batch, training=False)
        if predictions.shape[-1] is None:
            raise ValueError("Model output class dimension is unknown.")
        idx = int(class_index if class_index is not None else tf.argmax(predictions[0]).numpy())
        if idx < 0 or idx >= int(predictions.shape[-1]):
            raise ValueError("class_index is outside the model output range.")
        score = predictions[:, idx]
    grads = tape.gradient(score, conv_output)
    if grads is None:
        raise ValueError("Gradients are unavailable for this model architecture.")
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (maximum + tf.keras.backend.epsilon())
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (h, w)).numpy().squeeze()
    labels = None
    if labels_path and Path(labels_path).expanduser().is_file():
        labels = json.loads(Path(labels_path).expanduser().read_text(encoding="utf-8"))
    label = labels[idx] if isinstance(labels, list) and idx < len(labels) else f"class_{idx}"
    return {"heatmap": heatmap, "class_index": idx, "label": str(label),
            "score": float(predictions[0, idx].numpy()), "image": np.asarray(rgb),
            "model": path.name, "note": "Grad-CAM is approximate and can be misleading; do not treat it as causal proof."}


def evaluate_image_model(model_path: str, dataset_dir: str, batch_size: int = 16,
                         image_size: int = 160) -> dict[str, Any]:
    """Evaluate a saved classifier on class-named folders and return confusion matrix."""
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    model_file = Path(model_path).expanduser().resolve()
    root = Path(dataset_dir).expanduser().resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Model not found: {model_file}")
    if not root.is_dir():
        raise ValueError(f"Dataset folder not found: {root}")
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2:
        raise ValueError("Evaluation data must have at least two class-named subfolders.")
    ds = tf.keras.utils.image_dataset_from_directory(root, labels="inferred", label_mode="int",
        class_names=classes, shuffle=False, image_size=(int(image_size), int(image_size)), batch_size=int(batch_size))
    model = tf.keras.models.load_model(model_file)
    ys, ps = [], []
    for xb, yb in ds:
        xb = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(xb, tf.float32))
        pred = np.asarray(model.predict(xb, verbose=0))
        if pred.ndim != 2 or pred.shape[1] != len(classes):
            raise ValueError(f"Model output has {pred.shape[-1] if pred.ndim else 'unknown'} classes, but dataset has {len(classes)} folders.")
        ys.extend(np.asarray(yb).astype(int).tolist())
        ps.extend(pred.argmax(axis=1).astype(int).tolist())
    if not ys:
        raise ValueError("No images found in evaluation folders.")
    labels = list(range(len(classes)))
    matrix = confusion_matrix(ys, ps, labels=labels)
    report = classification_report(ys, ps, labels=labels, target_names=classes,
        output_dict=True, zero_division=0)
    return {"model_path": str(model_file), "dataset_path": str(root), "sample_count": len(ys),
        "accuracy": float(np.mean(np.asarray(ys) == np.asarray(ps))), "classes": classes,
        "confusion_matrix": matrix.tolist(), "classification_report": report,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def export_tflite(model_path: str, output_path: str, quantize: bool = False) -> dict[str, Any]:
    """Convert a saved Keras model to TensorFlow Lite; optional dynamic-range quantization."""
    import tensorflow as tf
    source = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Model not found: {source}")
    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() != ".tflite":
        raise ValueError("Output file must use the .tflite extension.")
    target.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(source)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    content = converter.convert()
    target.write_bytes(content)
    return {"source": str(source), "output": str(target), "size_bytes": len(content),
            "quantized": bool(quantize), "note": "Validate output parity and latency on the target device before deployment."}
