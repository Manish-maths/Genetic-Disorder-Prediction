# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Genetic Disorder Prediction", layout="wide")

# ===============================
# Load model and label encoder
# ===============================
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("genetic_disorder_pipeline.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("training_features.pkl")
    return pipeline, label_enc, feature_cols

pipeline, label_enc, feature_cols = load_pipeline()

# ===============================
# File upload
# ===============================
st.title("Genetic Disorder Prediction App")
st.write("Upload a CSV file with patient features (same columns as training data).")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.success(f"Uploaded file: {uploaded_file.name} ({df_test.shape[0]} rows)")

    # Drop irrelevant columns if exist
    drop_cols = [
        "Patient Id", "Family Name", "Institute Name", "Patient First Name",
        "Father's name", "Location of Institute", "Parental consent",
        "Test 1", "Test 2", "Test 3", "Test 4", "Test 5"
    ]
    df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors="ignore")

    # Align columns with training features
    for col in feature_cols:
        if col not in df_test.columns:
            df_test[col] = np.nan
    df_test = df_test[feature_cols]

    st.write("Preview of data:")
    st.dataframe(df_test.head())

    # ===============================
    # Predictions
    # ===============================
    probs = pipeline.predict_proba(df_test)
    top2_idx = np.argsort(probs, axis=1)[:, -2:]
    top2_probs = np.take_along_axis(probs, top2_idx, axis=1)
    top2_labels = np.array([label_enc.inverse_transform(row) for row in top2_idx])

    pred_top1 = top2_labels[:, 1]
    prob_top1 = top2_probs[:, 1]
    pred_top2 = top2_labels[:, 0]
    prob_top2 = top2_probs[:, 0]

    summary = [f"{p1} ({pr1:.2f}), 2nd: {p2} ({pr2:.2f})"
               for p1, pr1, p2, pr2 in zip(pred_top1, prob_top1, pred_top2, prob_top2)]

    df_results = df_test.copy()
    df_results["Prediction Top1"] = pred_top1
    df_results["Probability Top1"] = prob_top1
    df_results["Prediction Top2"] = pred_top2
    df_results["Probability Top2"] = prob_top2
    df_results["Prediction Summary"] = summary

    st.subheader("Predictions")
    st.dataframe(df_results[["Prediction Top1","Probability Top1","Prediction Top2","Probability Top2","Prediction Summary"]].head(10))

    # ===============================
    # Visualization
    # ===============================
    st.subheader("Visualizations")

    # Top-1 Class Distribution
    st.write("### Top-1 Prediction Class Distribution")
    plt.figure(figsize=(10,4))
    sns.countplot(
        x="Prediction Top1",
        data=df_results,
        order=np.sort(df_results["Prediction Top1"].unique()),
        palette="Set2"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Top-1 Probability Histogram
    st.write("### Top-1 Prediction Probability Histogram")
    plt.figure(figsize=(10,4))
    sns.histplot(df_results["Probability Top1"], bins=25, kde=True, color="#4c72b0", edgecolor='black')
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Low-confidence predictions
    st.write("### Low-Confidence Predictions (Top-1 Probability < 0.6)")
    low_conf = df_results[df_results["Probability Top1"] < 0.6]
    st.write(f"{len(low_conf)} samples detected")
    if not low_conf.empty:
        st.dataframe(low_conf[["Prediction Top1","Probability Top1"]].head(10))
        plt.figure(figsize=(10,4))
        sns.histplot(low_conf["Probability Top1"], bins=15, kde=True, color="#e74c3c", edgecolor='black')
        plt.xlabel("Top-1 Probability")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    # Borderline predictions
    st.write("### Borderline Predictions (Top-1 vs Top-2 difference < 0.1)")
    borderline = df_results[np.abs(df_results["Probability Top1"] - df_results["Probability Top2"]) < 0.1]
    st.write(f"{len(borderline)} samples detected")
    if not borderline.empty:
        st.dataframe(borderline[["Prediction Top1","Probability Top1","Prediction Top2","Probability Top2"]].head(10))
        plt.figure(figsize=(10,6))
        sns.scatterplot(
            x="Probability Top1",
            y="Probability Top2",
            hue="Prediction Top1",
            data=borderline,
            palette="Set2",
            s=100
        )
        plt.plot([0,1],[0,1], 'k--', label="Equal Probabilities")
        plt.xlabel("Top-1 Probability")
        plt.ylabel("Top-2 Probability")
        plt.legend(title="Top-1 Class")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()

    # Probability heatmap (first 50 samples)
    st.write("### Probability Heatmap (first 50 samples)")
    class_names = label_enc.classes_
    probs_df = pd.DataFrame(probs, columns=class_names).iloc[:50]
    plt.figure(figsize=(14,6))
    sns.heatmap(probs_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label':'Probability'})
    plt.xlabel("Classes")
    plt.ylabel("Samples")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
