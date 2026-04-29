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
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import load

# ── Setup ─────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
dataset   = pd.read_csv(file_path)
dataset   = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# ── AWS Secrets ───────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

MODEL_INFO = {
    "endpoint" : aws_endpoint,
    "explainer": "explainer_loan.shap",
    "pipeline" : "finalized_loan_model.tar.gz",
    "keys"     : ["int_rate", "fico_avg", "dti", "payment_to_income"],
    "inputs"   : [
        {"name": "int_rate",          "label": "Interest Rate (%)",        "min": 5.0,   "max": 30.0,  "default": 12.0, "step": 0.5},
        {"name": "fico_avg",          "label": "FICO Score",               "min": 600.0, "max": 850.0, "default": 700.0, "step": 5.0},
        {"name": "dti",               "label": "Debt-to-Income Ratio (%)", "min": 0.0,   "max": 40.0,  "default": 15.0, "step": 0.5},
        {"name": "payment_to_income", "label": "Payment-to-Income Ratio",  "min": 0.0,   "max": 1.0,   "default": 0.10, "step": 0.01},
    ]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)

def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping  = {0: "✅ Fully Paid (Low Risk)", 1: "⚠️ Default (High Risk)"}
        return mapping.get(pred_val, str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Apply imputer step only (smote and model are excluded)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df_transformed   = preprocessing_pipeline.transform(pd.DataFrame([input_df]))
    feature_names          = dataset.columns.tolist()
    input_df_transformed   = pd.DataFrame(input_df_transformed, columns=feature_names)

    # TreeExplainer returns a list [class_0, class_1] — we use class 1 (Default)
    shap_vals = explainer.shap_values(input_df_transformed)
    shap_vals_default = shap_vals[1][0]  # class 1, first row

    # Build Explanation object for waterfall plot
    exp = shap.Explanation(
        values      = shap_vals_default,
        base_values = explainer.expected_value[1],
        data        = input_df_transformed.iloc[0].values,
        feature_names = feature_names
    )

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(exp, max_display=12, show=False)
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_vals_default, index=feature_names)
        .abs().idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# ── Streamlit UI ──────────────────────────────────────────
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Prediction")
st.markdown(
    "Enter the borrower's key financial details. "
    "The model will predict whether this loan is likely to be **fully repaid** or result in a **default**."
)

with st.form("pred_form"):
    st.subheader("Borrower Financial Profile")
    cols        = st.columns(2)
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

original = dataset.iloc[0:1].to_dict(orient='records')[0]
original.update(user_inputs)

if submitted:
    with st.spinner("Calling model..."):
        res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        with st.spinner("Generating explanation..."):
            display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
