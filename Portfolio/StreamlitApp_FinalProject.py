import os
import sys
import warnings
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
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

import shap

# ============================================================
# SETUP
# ============================================================
warnings.simplefilter("ignore")

# ── AWS Secrets ───────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session (cached — same pattern as all HW apps) ────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ============================================================
# MODEL CONFIGURATION
# Top features by importance from the Tuned Random Forest.
# ============================================================
INPUT_FEATURES = [
    "int_rate",
    "term",
    "fico_avg",
    "dti",
    "inq_last_6mths",
    "payment_to_income",
]

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_loan.shap",
    "pipeline":  "finalized_loan_model.tar.gz",
    "inputs": [
        {"name": "int_rate",          "label": "Interest Rate (%)",             "min": 5.0,   "max": 30.0,  "default": 12.0, "step": 0.5},
        {"name": "term",              "label": "Loan Term (months)",            "min": 36.0,  "max": 60.0,  "default": 36.0, "step": 24.0},
        {"name": "fico_avg",          "label": "FICO Score",                   "min": 600.0, "max": 850.0, "default": 700.0, "step": 5.0},
        {"name": "dti",               "label": "Debt-to-Income Ratio (%)",      "min": 0.0,   "max": 40.0,  "default": 15.0, "step": 0.5},
        {"name": "inq_last_6mths",    "label": "Credit Inquiries (Last 6 Mo)", "min": 0.0,   "max": 10.0,  "default": 1.0,  "step": 1.0},
        {"name": "payment_to_income", "label": "Payment-to-Income Ratio",      "min": 0.0,   "max": 1.0,   "default": 0.1,  "step": 0.01},
    ]
}

# ============================================================
# LOAD PIPELINE FROM S3
# ============================================================
def load_pipeline(_session, bucket):
    s3_client = _session.client("s3")
    filename  = MODEL_INFO["pipeline"]
    s3_key    = f"sklearn-pipeline-deployment/{filename}"

    s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=filename)

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(joblib_file)

# ============================================================
# LOAD SHAP EXPLAINER FROM S3
# ============================================================
def load_shap_explainer(_session, bucket):
    s3_client      = _session.client("s3")
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)
    s3_key         = posixpath.join("explainer", explainer_name)

    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# ============================================================
# PREDICTION — calls live SageMaker endpoint
# ============================================================
def call_model_api(user_inputs: dict):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(user_inputs)
        pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping  = {0: "✅ FULLY PAID  (Low Risk)", 1: "⚠️ DEFAULT  (High Risk)"}
        return mapping.get(pred_val, str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ============================================================
# SHAP EXPLANATION
# ============================================================
def display_explanation(user_inputs: dict, session, bucket):
    input_df = pd.DataFrame([user_inputs], columns=INPUT_FEATURES)

    # Apply imputer step locally (same as HW6)
    pipeline = load_pipeline(session, bucket)
    imputer  = pipeline.named_steps["imputer"]
    X_imp    = imputer.transform(input_df)
    X_imp_df = pd.DataFrame(X_imp, columns=INPUT_FEATURES)

    explainer   = load_shap_explainer(session, bucket)
    shap_values = explainer(X_imp_df)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=6, show=False)
    st.pyplot(fig)

    # Most influential feature
    top_feature = (
        pd.Series(shap_values[0].values, index=shap_values[0].feature_names)
        .abs()
        .idxmax()
    )
    label_map = {
        "int_rate":          "Interest Rate",
        "term":              "Loan Term",
        "fico_avg":          "FICO Score",
        "dti":               "Debt-to-Income Ratio",
        "inq_last_6mths":    "Recent Credit Inquiries",
        "payment_to_income": "Payment-to-Income Ratio",
    }
    readable = label_map.get(top_feature, top_feature)
    st.info(f"**Business Insight:** The most influential factor in this prediction was **{readable}**.")

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Prediction")
st.markdown(
    "Enter the borrower's financial profile below. "
    "The model will predict whether the loan is likely to be "
    "**fully repaid** or result in a **default**."
)

with st.form("pred_form"):
    st.subheader("Borrower Financial Profile")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["label"],
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"]
            )

    submitted = st.form_submit_button("🔮 Run Prediction")

if submitted:
    with st.spinner("Calling model..."):
        res, status = call_model_api(user_inputs)

    if status == 200:
        st.metric("Prediction Result", res)
        with st.spinner("Generating explanation..."):
            display_explanation(user_inputs, session, aws_bucket)
    else:
        st.error(res)
