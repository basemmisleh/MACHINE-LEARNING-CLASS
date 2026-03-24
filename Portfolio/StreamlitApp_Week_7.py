import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap


# ── Setup & Path Configuration ────────────────────────────────────────────────
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair

# ── Access Secrets ────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session Management ────────────────────────────────────────────────────
@st.cache_resource  # Avoid re-creating the session on every page refresh
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Data & Model Configuration ────────────────────────────────────────────────
df_features = extract_features_pair()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_pair.shap",
    "pipeline":  "finalized_pair_model.tar.gz",
    "keys":   ["GOOGL", "ADI"],
    "inputs": [
        {"name": k, "type": "number", "min": 0.0, "default": 0.0, "step": 10.0}
        for k in ["GOOGL", "ADI"]
    ]
}

# ── Session State Initialization ──────────────────────────────────────────────
# FIX: Store prediction results in session_state so they survive
# the page rerun triggered when the user clicks the SHAP checkbox.
# Without this, clicking the checkbox resets the page because
# `submitted` becomes False on rerun and wipes the result block.
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None   # stores (res, status)
if "input_df" not in st.session_state:
    st.session_state.input_df = None            # stores the input DataFrame


# ── Helper: Load Pipeline from S3 ────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )

    # Extract the .joblib file from the .tar.gz archive
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(f"{joblib_file}")


# ── Helper: Load SHAP Explainer from S3 ──────────────────────────────────────
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client  = _session.client('s3')
    local_path = local_path

    # Only download if not already cached locally
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# ── Prediction Logic ──────────────────────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── Local Explainability (SHAP) ───────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline          = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df_transformed   = preprocessing_pipeline.transform(input_df)
    feature_names          = best_pipeline[1:4].get_feature_names_out()
    input_df_transformed   = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values            = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")

    # Class index 2 = BUY signal (0=SELL, 1=HOLD, 2=BUY)
    # Both the waterfall plot and top_feature MUST use the same index
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 2], max_display=10)
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 2].values, index=shap_values[0, :, 2].feature_names)
        .abs()
        .idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this BUY signal decision was **{top_feature}**.")


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pairs Trading Signal - GOOGL/ADI", layout="wide")
st.title("📈 Pairs Trading Signal — GOOGL / ADI")
st.markdown("Enter the current stock prices for the pair. The model will predict the trading signal for **GOOGL**.")

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

# ── On Form Submit: Run prediction and store in session_state ─────────────────
if submitted:
    input_df = pd.DataFrame(
        [[user_inputs[k] for k in MODEL_INFO["keys"]]],
        columns=MODEL_INFO["keys"]
    )
    # Save to session_state so results persist across reruns (e.g. checkbox click)
    st.session_state.input_df = input_df
    st.session_state.prediction_result = call_model_api(input_df.to_numpy())

# ── Display Results (reads from session_state, survives checkbox rerun) ───────
if st.session_state.prediction_result is not None:
    res, status = st.session_state.prediction_result

    if status == 200:
        signal_map   = {1.0: "BUY 🟢", 0.0: "HOLD 🟡", -1.0: "SELL 🔴"}
        signal_label = signal_map.get(res, str(res))
        st.metric("Trading Signal", signal_label)
        st.caption("Signal for GOOGL: -1 = SELL, 0 = HOLD, 1 = BUY (based on next-day return threshold +/-1%).")

        # FIX: Checkbox is now OUTSIDE the `if submitted` block.
        # It renders on every rerun because session_state holds the result.
        # Clicking it triggers a rerun but session_state still has the data
        # so the metric + checkbox + SHAP plot all render correctly.
        if st.checkbox("Show SHAP Explanation"):
            display_explanation(st.session_state.input_df, session, aws_bucket)

    else:
        st.error(res)
