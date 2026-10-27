# Modern learning track (2026)

This track supplements historical LiveLessons; it does not overwrite original lesson outputs. Maintainer: **Harpreet Singh**.

## Install

Use a fresh Python 3.11/3.12 environment. Install only the extras you need:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-modern.txt
python -m pip install -r requirements-ai.txt
```

## New AI features

### 1. Offline learning chatbot

```bash
python -m modern.chatbot
```

This is a small FAQ retrieval demo, not a generative LLM. It works offline and is easy to extend. Do not treat its responses as authoritative.

### 2. Transformer/NLP model scaffold

`modern/transformer_nlp.py` defines a Keras Transformer encoder and binary text-classification model. It prints a model summary, but does not download a dataset or claim trained accuracy. To train it, prepare integer token sequences padded to `max_length`, binary labels, and a train/validation split. Ensure train/test separation to avoid leakage.

### 3. Training dashboard

```bash
streamlit run modern/dashboard.py
```

Upload a CSV with an `epoch` column and metrics such as `loss`, `accuracy`, `val_loss`, and `val_accuracy`. For live TensorBoard event logs, add `TensorBoard(log_dir="logs")` to Keras callbacks and run:

```bash
tensorboard --logdir logs --host 127.0.0.1 --port 6006
```

### 4. Model API deployment

```bash
uvicorn modern.api:app --host 127.0.0.1 --port 8000
```

Try `GET /health` and `POST /chat` at `http://127.0.0.1:8000/docs`. For predictions, set `MODEL_PATH` to a compatible saved `.keras` model and send a `values` array matching that model's expected input shape to `POST /predict`. The generic endpoint is a demo; it does not automatically preprocess images or text.

**Security:** keep the development API bound to localhost. Before public deployment, add authentication, HTTPS/TLS termination, rate limits, request-size limits, logging/monitoring, model artifact controls, and secrets management. Never expose an unauthenticated development server directly to the internet.

## Existing modern vision lesson

Open `notebooks/modern_2026_cifar10_keras3.ipynb` for CIFAR-10, augmentation, callbacks, TensorBoard, evaluation and `.keras` save/reload. TensorFlow/GPU compatibility depends on OS and hardware; consult official install guidance. No training accuracy is guaranteed until the experiment is run.

## Extra MLOps and data tools

These additions use the Python standard library; they do not require a cloud account.

### 5. Dataset profiling and numeric drift screening

```bash
python -m modern.data_quality profile path/to/data.csv
python -m modern.data_quality drift reference.csv current.csv --bins 10
```

Profiling reports row/column counts, missing values, unique values and basic numeric statistics. Drift compares shared numeric columns using histogram total-variation distance; `screening_flag` is only a heuristic alert, not a statistical conclusion or evidence that model quality has changed. Use representative reference/current samples and domain-reviewed thresholds.

### 6. Experiment tracking and model cards

Append experiment metrics to a local JSONL file:

```bash
python -m modern.experiment_tracker --name baseline --metrics '{"accuracy":0.8,"loss":0.6}' --params '{"learning_rate":0.001}'
```

Python users can call `write_model_card(...)` to generate a Markdown model card. Only log metrics that were actually measured; the sample command is illustrative.

### 7. Local model registry with artifact integrity checks

```bash
python -m modern.model_registry register path/to/model.keras --name sentiment --version v1 --notes "first local build"
python -m modern.model_registry verify --name sentiment --version v1
```

The registry copies an artifact and stores a SHA-256 hash and metadata; verification detects accidental/tampered file changes. It does not load or execute the model, validate accuracy, manage access control, or replace a production registry. Keep registry directories private and back them up safely.

### Suggested next steps

For a production workflow, add CI checks, authentication, access-controlled artifact storage, automated evaluation gates, alerting, and a deployment platform only after the model's data pipeline and performance have been validated.

## Computer vision + AI Studio dashboard

Install the optional dashboard/system-monitor dependencies and the TensorFlow environment:

```bash
python -m pip install -r requirements-modern.txt
python -m pip install -r requirements-vision.txt
streamlit run modern/ai_dashboard.py
```

Features:
- Upload JPG/PNG/WEBP/BMP images and get top-k predictions from ImageNet MobileNetV2, or point the sidebar to a compatible saved `.keras` model. The first ImageNet run may download pretrained weights.
- Best-effort TensorFlow GPU device detection and NVIDIA `nvidia-smi` utilization/memory/temperature telemetry, plus optional CPU/RAM metrics via `psutil`.
- Confirmed, user-triggered transfer-learning retraining using labeled images in class-named folders, with validation split, early stopping, checkpoint, TensorBoard logs, class labels and a saved Keras model.
- Training history CSV/JSON charting and JSON prediction export.

Example labeled dataset layout:

```text
data/image_dataset/
  cats/one.jpg
  cats/two.jpg
  dogs/one.jpg
  dogs/two.jpg
```

Use a clean dataset with at least two classes and at least four images. Training uses MobileNetV2 pretrained weights, so a first run may need internet access to download weights. Training may be slow on CPU. The retraining button is not an unattended scheduled retraining service; it requires local user confirmation. GPU statistics are best-effort snapshots and may be unavailable without compatible drivers and `nvidia-smi`. This dashboard is a local educational prototype; predictions can be wrong. Do not expose it to the public internet without authentication, request limits, upload limits and deployment security review.

### Extra AI Studio features

The web dashboard also provides:

- **Batch Prediction:** upload multiple images and export prediction results as CSV/JSON. An invalid image is reported individually without stopping the rest of the batch.
- **Dataset Audit:** inspect class-folder counts, image extensions, approximate size, unreadable files, and a simple class-imbalance ratio before training.
- **Training augmentation:** transfer-learning training applies random horizontal flips, rotations and zoom during training only.

Open the dashboard with `streamlit run modern/ai_dashboard.py`. Use `data/image_dataset/<class_name>/...` for labeled training data. The audit is heuristic and does not replace a separate test set or manual label review.

### Additional AI Studio tools

The `modern/ai_dashboard.py` dashboard now includes:

- Grad-CAM heatmaps for compatible custom convolutional models.
- Evaluation on a separate labeled test folder with confusion matrix and precision/recall/F1.
- TensorFlow Lite export, optionally using dynamic-range quantization.
- A local JSONL prediction audit log (metadata only; it does not store uploaded image bytes).

Install `requirements-vision.txt` for these optional tools. The evaluation folder should have one subfolder per class, matching the saved model's output class order. Use a held-out test set. Grad-CAM can fail on architectures with disconnected/nested convolutional graphs and should be treated as an approximate visualization, not a causal explanation. Validate converted TFLite models on the target hardware before shipping.


## Additional AI Studio utilities

The dashboard also includes reproducible dataset split manifests, a single-image Keras inference benchmark, and a static deployment-readiness checklist. Open the **Dataset Splits**, **Inference Benchmark**, and **Deployment Check** tabs. The split tool only writes a JSON manifest; it does not move/copy files. For datasets with multiple frames from the same source, create grouped splits to prevent leakage. Benchmark figures depend on hardware and model/runtime settings. The readiness checklist is advisory and does not replace authentication, HTTPS, rate limiting, privacy review, dependency scanning, or load testing.
