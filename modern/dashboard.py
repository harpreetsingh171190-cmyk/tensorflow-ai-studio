"""Lightweight Streamlit dashboard for TensorBoard event logs or exported training-history CSV."""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Harpreet Singh | ML Training Dashboard", page_icon="📈", layout="wide")
st.title("ML Live Training Dashboard")
st.caption("Educational monitor for exported Keras history CSV files. For live event streaming, open TensorBoard as described below.")
upload = st.file_uploader("Upload history.csv (for example, exported from keras.callbacks.History)", type=["csv"])
if upload is not None:
    df = pd.read_csv(upload)
    st.write("Available metrics", list(df.columns))
    metric_cols = [c for c in df.columns if c.lower() not in {"epoch", "step", "batch"} and pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        st.warning("No numeric metric columns found in this CSV.")
    else:
        selected = st.multiselect("Metrics to chart", metric_cols, default=metric_cols[:min(4, len(metric_cols))])
        x = "epoch" if "epoch" in df.columns else ("step" if "step" in df.columns else None)
        chart_data = df.set_index(x)[selected] if x and selected else df[selected]
        if selected:
            st.line_chart(chart_data)
        st.dataframe(df.tail(20), use_container_width=True)
else:
    st.info("No CSV uploaded yet. Save `history.history` to CSV from your training notebook, then upload it here.")

st.subheader("TensorBoard live logs")
st.code("tensorboard --logdir logs --host 127.0.0.1 --port 6006", language="bash")
st.markdown("Open `http://127.0.0.1:6006` locally. Keep dashboards on localhost unless you have configured authentication and network controls.")
