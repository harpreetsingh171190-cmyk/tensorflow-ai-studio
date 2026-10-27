"""Modern local Streamlit dashboard for image prediction, GPU telemetry and retraining."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from modern.computer_vision import gpu_status, predict_image, retrain_image_classifier
from modern.vision_extras import audit_image_dataset, batch_predict_images
from modern.model_tools import gradcam_heatmap, evaluate_image_model, export_tflite
from modern.prediction_log import append_prediction_log, read_prediction_log
from modern.feedback_queue import add_feedback, read_feedback
from modern.model_monitoring import summarize_predictions, compare_prediction_windows
from modern.vision_ops import build_split_manifest, deployment_readiness, benchmark_keras_model
from modern.dataset_tools import find_duplicate_images, find_split_leakage
from modern.metrics import classification_metrics, expected_calibration_error
from modern.experiment_tracker import read_experiments, best_experiment

st.set_page_config(page_title="Harpreet Singh | AI Studio", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1440px;}
.hero {padding: 1.35rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#14213d,#234e70); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {opacity: .88; margin-bottom: 0;}
</style>
<div class="hero"><h1>🧠 Harpreet Singh · AI Studio</h1><p>Computer vision · GPU monitoring · experiment feedback · controlled model retraining</p></div>
""", unsafe_allow_html=True)

model_path = st.sidebar.text_input("Optional custom .keras model path", value=os.getenv("MODEL_PATH", ""))
labels_path = st.sidebar.text_input("Optional class_names.json path", value=os.getenv("LABELS_PATH", ""))
st.sidebar.caption("If no custom model is selected, image predictions use pretrained ImageNet MobileNetV2 (first run may download weights).")

predict_tab, batch_tab, audit_tab, monitor_tab, train_tab, history_tab, explain_tab, eval_tab, tools_tab, log_tab, feedback_tab, model_monitor_tab, split_tab, benchmark_tab, deploy_tab, safety_tab, metrics_tab, experiments_tab = st.tabs(["🖼️ Image Prediction", "📦 Batch Prediction", "🧪 Dataset Audit", "⚡ GPU & System", "🔁 Retrain Model", "📈 Training History", "🔎 Explainability", "🎯 Model Evaluation", "📱 Model Export", "🧾 Prediction Log", "🧑‍⚖️ Human Review", "📊 Model Monitoring", "🗂️ Dataset Splits", "⏱️ Inference Benchmark", "🚀 Deployment Check", "🛡️ Dataset Safety", "📐 Metrics & Calibration", "🧬 Experiments"])

with predict_tab:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="cv_upload")
        top_k = st.slider("Top predictions", min_value=1, max_value=10, value=5)
        run_prediction = st.button("Analyze image", type="primary", disabled=upload is None, use_container_width=True)
    with right:
        if upload is not None:
            try:
                image = Image.open(upload).convert("RGB")
                st.image(image, caption=f"{upload.name} · {image.width} × {image.height}", use_container_width=True)
            except Exception:
                st.error("This image could not be previewed. Please upload a valid image file.")
    if run_prediction and upload is not None:
        with st.spinner("Running image model…"):
            try:
                result = predict_image(upload.getvalue(), model_path.strip() or None, labels_path.strip() or None, top_k)
                log_path = append_prediction_log(result, upload.name)
                st.session_state["latest_prediction_review"] = {"filename": upload.name, "model": result.get("model", ""), "predictions": result.get("predictions", [])}
                st.caption(f"Prediction metadata saved locally: {log_path}")
                st.subheader(f"Predictions · {result['model']}")
                result_df = pd.DataFrame(result["predictions"])
                result_df["confidence_percent"] = (result_df["confidence"] * 100).round(2)
                st.dataframe(result_df[["label", "class_id", "confidence_percent"]], hide_index=True, use_container_width=True)
                st.bar_chart(result_df.set_index("label")["confidence"])
                st.info(result["warning"])
                st.download_button("Download prediction JSON", json.dumps(result, indent=2), file_name="image_prediction.json", mime="application/json")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with batch_tab:
    st.subheader("Batch image prediction")
    batch_uploads = st.file_uploader("Select multiple images", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=True, key="batch_images")
    batch_k = st.slider("Top predictions per image", 1, 10, 3, key="batch_top_k")
    if st.button("Run batch prediction", type="primary", disabled=not batch_uploads, use_container_width=True):
        with st.spinner(f"Analyzing {len(batch_uploads)} image(s)…"):
            batch_result = batch_predict_images([(f.name, f.getvalue()) for f in batch_uploads], model_path.strip() or None, labels_path.strip() or None, batch_k)
        st.session_state["batch_prediction_result"] = batch_result
    batch_result = st.session_state.get("batch_prediction_result")
    if batch_result:
        b1, b2, b3 = st.columns(3)
        b1.metric("Files", batch_result["processed"])
        b2.metric("Succeeded", batch_result["succeeded"])
        b3.metric("Failed", batch_result["failed"])
        batch_df = pd.DataFrame(batch_result["results"])
        st.dataframe(batch_df, hide_index=True, use_container_width=True)
        st.download_button("Download batch results CSV", batch_df.to_csv(index=False).encode("utf-8"), file_name="batch_predictions.csv", mime="text/csv")
        st.download_button("Download batch results JSON", json.dumps(batch_result, indent=2, ensure_ascii=False), file_name="batch_predictions.json", mime="application/json")
        if "error" in batch_df.columns and batch_df["error"].notna().any():
            st.warning("Some files could not be processed. Review the error column; other predictions are preserved.")

with audit_tab:
    st.subheader("Labeled image dataset quality audit")
    st.markdown("Expected layout: `dataset/class_a/image.jpg` and `dataset/class_b/image.jpg`. This audit reads metadata and verifies image files; it does not upload data anywhere.")
    audit_path = st.text_input("Dataset folder to audit", value="data/image_dataset", key="audit_path")
    if st.button("Audit dataset", type="primary", use_container_width=True):
        try:
            st.session_state["dataset_audit"] = audit_image_dataset(audit_path)
        except Exception as exc:
            st.error(f"Dataset audit failed: {exc}")
    audit = st.session_state.get("dataset_audit")
    if audit:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Classes", audit["class_count"])
        q2.metric("Images", audit["image_count"])
        q3.metric("Dataset size", f"{audit['total_size_mb']} MB")
        q4.metric("Imbalance ratio", str(audit["imbalance_ratio"] or "N/A"))
        class_df = pd.DataFrame(audit["classes"])
        st.bar_chart(class_df.set_index("class_name")["image_count"] if not class_df.empty else pd.Series(dtype=int))
        st.dataframe(class_df, hide_index=True, use_container_width=True)
        st.write("**Recommendations**")
        for tip in audit["recommendations"]:
            st.write(f"- {tip}")
        if audit["unreadable_images"]:
            st.error(f"Unreadable images ({len(audit['unreadable_images'])})")
            st.code("\n".join(audit["unreadable_images"][:100]))
        st.download_button("Download dataset audit JSON", json.dumps(audit, indent=2), file_name="dataset_audit.json", mime="application/json")

with monitor_tab:
    st.subheader("Runtime health")
    gpu = gpu_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("TensorFlow GPU", "Available" if gpu.get("available") else "Not detected")
    c2.metric("GPU devices", len(gpu.get("devices", [])))
    c3.metric("TensorFlow", gpu.get("tensorflow_version", "Not installed"))
    if gpu.get("devices"):
        st.write("Detected devices", gpu["devices"])
    if gpu.get("nvidia_smi"):
        st.subheader("NVIDIA telemetry")
        st.dataframe(pd.DataFrame(gpu["nvidia_smi"]), hide_index=True, use_container_width=True)
    else:
        st.warning("nvidia-smi telemetry is unavailable. This is normal on CPU-only systems or when GPU drivers/tools are not exposed to this process.")
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.25)
        m1, m2, m3 = st.columns(3)
        m1.metric("CPU usage", f"{cpu:.1f}%")
        m2.metric("RAM used", f"{mem.percent:.1f}%")
        m3.metric("RAM available", f"{mem.available / (1024**3):.2f} GB")
    except ImportError:
        st.caption("Install psutil to display CPU/RAM metrics.")
    st.caption("Telemetry is a point-in-time snapshot, not a continuously refreshing system service.")

with train_tab:
    st.subheader("Transfer-learning retraining")
    st.markdown("Put labeled images into **one subfolder per class**, for example `dataset/cats/*.jpg` and `dataset/dogs/*.jpg`. Keep only trusted datasets in this local directory.")
    dataset_dir = st.text_input("Local dataset folder", value="data/image_dataset")
    output_dir = st.text_input("Output folder", value="artifacts/vision")
    a, b = st.columns(2)
    epochs = a.slider("Maximum epochs", 1, 30, 5)
    batch_size = b.selectbox("Batch size", [8, 16, 32, 64], index=1)
    confirm = st.checkbox("I confirm this is a labeled training dataset and want to start training.")
    if st.button("Start retraining", type="primary", disabled=not confirm, use_container_width=True):
        progress = st.progress(0, text="Validating dataset and initializing training…")
        try:
            with st.spinner("Training can take several minutes. Do not close this page."):
                result = retrain_image_classifier(dataset_dir, output_dir, epochs=epochs, batch_size=batch_size)
            progress.progress(100, text="Training completed")
            st.success("Training finished. The model and class labels have been saved.")
            st.code(f"Model: {result['model_path']}\nLabels: {result['labels_path']}")
            st.json({"classes": result["classes"], "images": result["image_count"], "final_metrics": result["final_metrics"]})
            hist = pd.DataFrame(result["history"])
            if not hist.empty:
                hist.index.name = "epoch"
                st.line_chart(hist)
            st.info("To predict with this model, copy the Model and Labels paths into the sidebar fields and upload an image.")
        except Exception as exc:
            progress.empty()
            st.error(f"Retraining failed: {exc}")

with history_tab:
    st.subheader("Explore exported training metrics")
    history_upload = st.file_uploader("Upload history CSV or JSON", type=["csv", "json"], key="history_upload")
    if history_upload is not None:
        try:
            if history_upload.name.lower().endswith(".csv"):
                hist_df = pd.read_csv(history_upload)
            else:
                payload = json.load(history_upload)
                hist_df = pd.DataFrame(payload.get("history", payload))
            numeric = [c for c in hist_df.columns if pd.api.types.is_numeric_dtype(hist_df[c])]
            if numeric:
                chosen = st.multiselect("Metrics", numeric, default=numeric[:min(4, len(numeric))])
                if chosen:
                    st.line_chart(hist_df[chosen])
            st.dataframe(hist_df.tail(50), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read history file: {exc}")
    else:
        st.caption("Upload a Keras History CSV or the history.json file created by this dashboard.")


with explain_tab:
    st.subheader("Grad-CAM visual explanation")
    st.markdown("Create a heatmap showing image regions that may influence a CNN prediction. Requires a compatible custom `.keras` convolutional model; default ImageNet mode is not used here.")
    explain_upload = st.file_uploader("Upload an image for explanation", type=["jpg", "jpeg", "png", "webp", "bmp"], key="gradcam_upload")
    explain_model = st.text_input("Custom model path", value=model_path, key="gradcam_model")
    explain_labels = st.text_input("Optional class_names.json path", value=labels_path, key="gradcam_labels")
    if st.button("Generate Grad-CAM", disabled=not (explain_upload and explain_model.strip()), type="primary"):
        try:
            with st.spinner("Computing Grad-CAM…"):
                explanation = gradcam_heatmap(explain_upload.getvalue(), explain_model.strip(), explain_labels.strip() or None)
            import numpy as np
            from PIL import Image as PILImage
            original = explanation["image"].astype(np.float32)
            heat = explanation["heatmap"]
            # Simple red/yellow overlay, generated locally without writing the uploaded image to disk.
            color = np.zeros((*heat.shape, 3), dtype=np.float32)
            color[..., 0] = np.clip(heat * 2.0, 0, 1) * 255
            color[..., 1] = np.clip((heat - 0.35) * 1.7, 0, 1) * 180
            overlay = np.clip(original * 0.58 + color * 0.42, 0, 255).astype(np.uint8)
            c1, c2 = st.columns(2)
            c1.image(original.astype(np.uint8), caption="Original", use_container_width=True)
            c2.image(overlay, caption=f"Grad-CAM · {explanation['label']}", use_container_width=True)
            st.metric("Selected class score", f"{explanation['score']:.3f}")
            st.warning(explanation["note"])
        except Exception as exc:
            st.error(f"Grad-CAM failed: {exc}")

with eval_tab:
    st.subheader("Evaluate a saved model on labeled test images")
    st.markdown("Use a separate held-out test folder with one subfolder per class. The folder names must match the model's class order.")
    eval_model_path = st.text_input("Model file (.keras)", value=model_path, key="eval_model")
    eval_dataset = st.text_input("Test dataset folder", value="data/image_test", key="eval_dataset")
    eval_size = st.selectbox("Image size", [96, 128, 160, 192, 224], index=2, key="eval_size")
    if st.button("Run evaluation", type="primary", disabled=not eval_model_path.strip()):
        try:
            with st.spinner("Evaluating held-out images…"):
                report = evaluate_image_model(eval_model_path.strip(), eval_dataset.strip(), image_size=eval_size)
            st.session_state["evaluation_report"] = report
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
    report = st.session_state.get("evaluation_report")
    if report:
        e1, e2 = st.columns(2)
        e1.metric("Test images", report["sample_count"])
        e2.metric("Accuracy", f"{report['accuracy']:.1%}")
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(report["confusion_matrix"], index=report["classes"], columns=report["classes"]), use_container_width=True)
        report_df = pd.DataFrame(report["classification_report"]).T
        st.dataframe(report_df.round(4), use_container_width=True)
        st.download_button("Download evaluation JSON", json.dumps(report, indent=2), file_name="model_evaluation.json", mime="application/json")
        st.caption("Evaluation is only meaningful when test data was not used for training or tuning.")

with tools_tab:
    st.subheader("Export a trained model for edge/mobile use")
    st.markdown("Convert a saved Keras model into TensorFlow Lite (`.tflite`). Optional dynamic-range quantization may reduce size but must be validated for accuracy on the target device.")
    export_source = st.text_input("Source .keras model", value=model_path, key="export_source")
    export_target = st.text_input("Output file", value="artifacts/vision/model.tflite", key="export_target")
    quantize = st.checkbox("Enable dynamic-range quantization", value=False)
    if st.button("Convert to TensorFlow Lite", type="primary", disabled=not export_source.strip()):
        try:
            with st.spinner("Converting model…"):
                export_result = export_tflite(export_source.strip(), export_target.strip(), quantize=quantize)
            st.success(f"Exported model: {export_result['output']}")
            st.json(export_result)
        except Exception as exc:
            st.error(f"Export failed: {exc}")

with log_tab:
    st.subheader("Local prediction audit log")
    st.caption("Only filename, model name, top label, score and timestamp are stored; image bytes are not saved. The log is local to this machine.")
    log_file = st.text_input("Log path", value="artifacts/prediction_log.jsonl", key="log_path")
    rows = read_prediction_log(log_file, limit=1000)
    if rows:
        log_df = pd.DataFrame(rows)
        st.dataframe(log_df, use_container_width=True)
        st.download_button("Download audit log CSV", log_df.to_csv(index=False).encode("utf-8"), file_name="prediction_audit_log.csv", mime="text/csv")
    else:
        st.info("No prediction entries found at this path yet. Run an image prediction first.")

st.divider()
st.caption("Educational prototype · predictions may be wrong · retraining is manual-confirmation driven · keep the dashboard on localhost unless secured.")


with feedback_tab:
    st.subheader("Human-in-the-loop review queue")
    st.markdown("Correct a prediction to create a small metadata-only review record. These corrections are **not automatically used for training**; inspect labels and build a curated dataset before retraining.")
    latest = st.session_state.get("latest_prediction_review", {})
    latest_preds = latest.get("predictions", [])
    if latest_preds:
        default_pred = latest_preds[0]
        st.caption(f"Latest image: {Path(latest.get('filename', 'image')).name} · predicted {default_pred.get('label', 'unknown')}")
        review_label = st.text_input("Correct class label", key="review_corrected_label")
        review_notes = st.text_area("Optional review notes", max_chars=1000, key="review_notes")
        feedback_path = st.text_input("Feedback queue path", value="artifacts/feedback_queue.jsonl", key="feedback_path")
        if st.button("Save human correction", type="primary", disabled=not review_label.strip()):
            try:
                out = add_feedback(filename=latest.get("filename", "image"), model=latest.get("model", ""),
                    predicted_label=str(default_pred.get("label", "unknown")), confidence=default_pred.get("confidence"),
                    corrected_label=review_label, notes=review_notes, queue_path=feedback_path)
                st.success(f"Correction saved locally: {out}")
            except Exception as exc:
                st.error(f"Could not save correction: {exc}")
    else:
        st.info("Run an image prediction first. Its metadata will appear here for correction.")
    queue_path_view = st.text_input("Review queue to inspect", value="artifacts/feedback_queue.jsonl", key="feedback_view_path")
    feedback_rows = read_feedback(queue_path_view, limit=1000)
    if feedback_rows:
        feedback_df = pd.DataFrame(feedback_rows)
        st.dataframe(feedback_df, hide_index=True, use_container_width=True)
        st.download_button("Download feedback CSV", feedback_df.to_csv(index=False).encode("utf-8"), file_name="human_review_feedback.csv", mime="text/csv")
    else:
        st.caption("No human corrections saved yet.")

with model_monitor_tab:
    st.subheader("Prediction quality signals")
    st.markdown("Uses local prediction metadata only. This is a heuristic signal dashboard, not a validated statistical drift detector.")
    monitor_path = st.text_input("Prediction log path", value="artifacts/prediction_log.jsonl", key="monitor_log_path")
    monitor_rows = read_prediction_log(monitor_path, limit=10000)
    threshold = st.slider("Low-confidence threshold", 0.05, 0.95, 0.60, 0.05)
    summary = summarize_predictions(monitor_rows, confidence_threshold=threshold)
    m1, m2, m3 = st.columns(3)
    m1.metric("Logged predictions", summary["sample_count"])
    m2.metric("Mean top confidence", f"{summary['mean_confidence']:.1%}" if summary["mean_confidence"] is not None else "N/A")
    m3.metric("Low-confidence rate", f"{summary['low_confidence_rate']:.1%}" if summary["low_confidence_rate"] is not None else "N/A")
    if summary["label_counts"]:
        st.bar_chart(pd.Series(summary["label_counts"], name="count"))
    if len(monitor_rows) >= 10:
        split_at = max(1, len(monitor_rows) // 2)
        comparison = compare_prediction_windows(monitor_rows[:split_at], monitor_rows[split_at:], threshold)
        st.subheader("Older half vs newer half")
        d1, d2 = st.columns(2)
        d1.metric("Label-mix distance", f"{comparison['label_mix_distance']:.3f}", help="0 means same label mix; 1 means completely disjoint observed label mixes.")
        delta = comparison["confidence_delta"]
        d2.metric("Mean confidence change", f"{delta:+.1%}" if delta is not None else "N/A")
        st.warning(comparison["warning"])
    else:
        st.info("At least 10 logged predictions are recommended before comparing older and newer windows.")
    st.download_button("Download monitoring summary JSON", json.dumps(summary, indent=2), file_name="prediction_monitoring_summary.json", mime="application/json")


with split_tab:
    st.subheader("Reproducible train / validation / test manifest")
    st.markdown("Create a deterministic JSON manifest from class-named image folders. This does not copy or modify image files.")
    split_dataset = st.text_input("Dataset folder", value="data/image_dataset", key="split_dataset")
    c1, c2, c3 = st.columns(3)
    train_ratio = c1.number_input("Train ratio", min_value=0.10, max_value=0.90, value=0.70, step=0.05, key="split_train")
    val_ratio = c2.number_input("Validation ratio", min_value=0.05, max_value=0.45, value=0.15, step=0.05, key="split_val")
    test_ratio = c3.number_input("Test ratio", min_value=0.05, max_value=0.45, value=0.15, step=0.05, key="split_test")
    split_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, key="split_seed")
    split_out = st.text_input("Manifest output path", value="artifacts/vision/split_manifest.json", key="split_out")
    if st.button("Build split manifest", type="primary"):
        try:
            manifest = build_split_manifest(split_dataset, split_out, train_ratio, val_ratio, test_ratio, int(split_seed))
            st.success(f"Manifest saved: {split_out}")
            st.metric("Images indexed", manifest["sample_count"])
            st.dataframe(pd.DataFrame(manifest["class_counts"]).T, use_container_width=True)
            st.download_button("Download manifest JSON", json.dumps(manifest, indent=2), file_name="split_manifest.json", mime="application/json")
        except Exception as exc:
            st.error(f"Could not build manifest: {exc}")
    st.warning("For video frames, multiple images of the same scene or subject must remain in one split to avoid data leakage; this file-level splitter cannot infer groups.")

with benchmark_tab:
    st.subheader("Single-image inference benchmark")
    st.markdown("Measure latency for an existing custom Keras model after warm-up. Model loading and image decoding are excluded from the timed region.")
    bench_model = st.text_input("Model file (.keras)", value=model_path, key="bench_model")
    bench_image = st.file_uploader("Benchmark image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="bench_image")
    b1, b2 = st.columns(2)
    bench_repeats = b1.slider("Timed inference repeats", 1, 100, 10, key="bench_repeats")
    bench_warmup = b2.slider("Warm-up runs", 0, 10, 2, key="bench_warmup")
    if st.button("Run benchmark", type="primary", disabled=not (bench_model.strip() and bench_image)):
        try:
            with st.spinner("Loading model and benchmarking inference…"):
                bench_result = benchmark_keras_model(bench_model.strip(), bench_image.getvalue(), bench_repeats, bench_warmup)
            st.session_state["inference_benchmark"] = bench_result
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")
    bench_result = st.session_state.get("inference_benchmark")
    if bench_result:
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Mean latency", f"{bench_result['mean_latency_ms']:.2f} ms")
        x2.metric("Median latency", f"{bench_result['median_latency_ms']:.2f} ms")
        x3.metric("P95 latency", f"{bench_result['p95_latency_ms']:.2f} ms")
        x4.metric("Throughput", f"{bench_result['throughput_images_per_second']:.2f} images/s")
        st.json(bench_result)
        st.download_button("Download benchmark JSON", json.dumps(bench_result, indent=2), file_name="inference_benchmark.json", mime="application/json")
        st.caption(bench_result["note"])

with deploy_tab:
    st.subheader("Model deployment readiness checklist")
    st.markdown("Static artifact and configuration checks for a saved model. This is not a security audit and does not certify production readiness.")
    deploy_model = st.text_input("Model artifact path", value=model_path, key="deploy_model")
    deploy_labels = st.text_input("Class labels JSON (recommended)", value=labels_path, key="deploy_labels")
    if st.button("Run readiness checks", type="primary"):
        try:
            readiness = deployment_readiness(deploy_model.strip(), deploy_labels.strip() or None)
            st.session_state["deployment_readiness"] = readiness
        except Exception as exc:
            st.error(f"Readiness check failed: {exc}")
    readiness = st.session_state.get("deployment_readiness")
    if readiness:
        r1, r2 = st.columns(2)
        r1.metric("Checklist score", f"{readiness['readiness_percent']}%")
        r2.metric("Passed checks", f"{readiness['passed_checks']} / {readiness['total_checks']}")
        st.dataframe(pd.DataFrame(readiness["checks"]), hide_index=True, use_container_width=True)
        st.warning(readiness["disclaimer"])
        st.download_button("Download readiness JSON", json.dumps(readiness, indent=2), file_name="deployment_readiness.json", mime="application/json")


with safety_tab:
    st.subheader("Dataset safety: duplicates and split leakage")
    st.markdown("Finds **byte-identical** images (also the same picture filed under different classes) and identical images that a split manifest put in different splits. Near-duplicates are not detected.")
    dup_path = st.text_input("Dataset folder", value="data/image_dataset", key="dup_path")
    if st.button("Find duplicates", type="primary", key="dup_run"):
        try:
            st.session_state["dup_result"] = find_duplicate_images(dup_path.strip())
        except Exception as exc:
            st.error(f"Duplicate check failed: {exc}")
    dup = st.session_state.get("dup_result")
    if dup:
        d1, d2, d3 = st.columns(3)
        d1.metric("Images scanned", dup["image_count"])
        d2.metric("Redundant files", dup["redundant_file_count"])
        d3.metric("Cross-class conflicts", dup["cross_class_groups"])
        if dup["cross_class_groups"]:
            st.error("The same image appears under different classes. Fix these labels before training.")
        if dup["duplicate_groups"]:
            st.dataframe(pd.DataFrame([{"copies": g["count"], "cross_class": g["cross_class"], "files": ", ".join(g["files"])} for g in dup["duplicate_groups"]]), hide_index=True, use_container_width=True)
        else:
            st.success("No byte-identical duplicates found.")
        st.download_button("Download duplicate report JSON", json.dumps(dup, indent=2), file_name="duplicate_report.json", mime="application/json")
    st.divider()
    manifest_path = st.text_input("Split manifest JSON", value="artifacts/vision/split_manifest.json", key="leak_path")
    if st.button("Check split leakage", key="leak_run"):
        try:
            st.session_state["leak_result"] = find_split_leakage(manifest_path.strip())
        except Exception as exc:
            st.error(f"Leakage check failed: {exc}")
    leak = st.session_state.get("leak_result")
    if leak:
        l1, l2, l3 = st.columns(3)
        l1.metric("Files checked", leak["checked_files"])
        l2.metric("Leaked files", leak["leaked_file_count"])
        l3.metric("Missing files", leak["missing_files"])
        if leak["leaking_groups"]:
            st.error("Identical images appear in more than one split, so test metrics would be inflated.")
            st.json(leak["leaking_groups"])
        else:
            st.success("No exact-duplicate leakage between splits.")
        st.caption(leak["note"])

with metrics_tab:
    st.subheader("Classification metrics and calibration")
    st.markdown("Upload a CSV with true labels and predicted labels (and optionally a confidence column between 0 and 1).")
    metrics_file = st.file_uploader("Predictions CSV", type=["csv"], key="metrics_csv")
    if metrics_file is not None:
        try:
            mdf = pd.read_csv(metrics_file)
            cols = list(mdf.columns)
            m1, m2, m3 = st.columns(3)
            true_col = m1.selectbox("True label column", cols, index=0)
            pred_col = m2.selectbox("Predicted label column", cols, index=min(1, len(cols) - 1))
            conf_col = m3.selectbox("Confidence column (optional)", ["(none)"] + cols)
            if st.button("Compute metrics", type="primary", key="metrics_run"):
                truth, pred = mdf[true_col].astype(str).tolist(), mdf[pred_col].astype(str).tolist()
                report = classification_metrics(truth, pred)
                st.session_state["metrics_report"] = report
                st.session_state["calibration_report"] = None
                if conf_col != "(none)":
                    conf = pd.to_numeric(mdf[conf_col], errors="coerce")
                    ok = conf.notna()
                    correct = [t == p for t, p, keep in zip(truth, pred, ok) if keep]
                    st.session_state["calibration_report"] = expected_calibration_error(conf[ok].tolist(), correct)
        except Exception as exc:
            st.error(f"Metrics failed: {exc}")
    report = st.session_state.get("metrics_report")
    if report:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{report['accuracy']:.3f}")
        k2.metric("Macro F1", f"{report['macro_f1']:.3f}")
        k3.metric("Weighted F1", f"{report['weighted_f1']:.3f}")
        k4.metric("Samples", report["sample_count"])
        st.dataframe(pd.DataFrame(report["per_class"]).T, use_container_width=True)
        cm = report["confusion_matrix"]
        st.write("Confusion matrix (rows = true, columns = predicted)")
        st.dataframe(pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]), use_container_width=True)
        st.download_button("Download metrics JSON", json.dumps(report, indent=2), file_name="classification_metrics.json", mime="application/json")
    calibration = st.session_state.get("calibration_report")
    if calibration:
        st.metric("Expected calibration error", f"{calibration['ece']:.4f}")
        st.dataframe(pd.DataFrame(calibration["bins"]), hide_index=True, use_container_width=True)
        st.caption("Lower is better. The value depends on the number of bins and sample size; treat small samples with caution.")

with experiments_tab:
    st.subheader("Experiment log")
    runs_path = st.text_input("Experiment log (JSONL)", value="experiments/runs.jsonl", key="runs_path")
    try:
        runs = read_experiments(runs_path.strip())
    except Exception as exc:
        runs = []
        st.error(f"Could not read the log: {exc}")
    if not runs:
        st.info("No runs found. Log runs with modern.experiment_tracker.log_experiment(...).")
    else:
        st.dataframe(pd.DataFrame([{"time": r.get("timestamp_utc"), "name": r.get("name"), **r.get("metrics", {})} for r in runs]), hide_index=True, use_container_width=True)
        metric_names = sorted({k for r in runs for k in r.get("metrics", {})})
        if metric_names:
            e1, e2 = st.columns(2)
            chosen_metric = e1.selectbox("Metric", metric_names)
            chosen_mode = e2.radio("Best means", ["max", "min"], horizontal=True)
            best = best_experiment(chosen_metric, chosen_mode, runs_path.strip())
            if best:
                st.success(f"Best run: {best.get('name')} ({chosen_metric} = {best['metrics'][chosen_metric]})")
                st.json(best)
