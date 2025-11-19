import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# -------------------------
# Load your dataset
# -------------------------
df = pd.read_csv("genetic_data.csv")

# Drop irrelevant columns
drop_cols = [
    "Patient Id", "Family Name", "Institute Name", "Patient First Name",
    "Father's name", "Location of Institute", "Parental consent",
    "Test 1", "Test 2", "Test 3", "Test 4", "Test 5"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Target
y = df["Genetic Disorder"]
X = df.drop(columns=["Genetic Disorder"])

# Encode target
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

# Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# Model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss"
)

# Full pipeline
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# Train
pipeline.fit(X, y_enc)

# Save pipeline + encoder + feature list
joblib.dump(pipeline, "genetic_disorder_pipeline.pkl")
joblib.dump(label_enc, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "training_features.pkl")

print("Training completed and files saved!")
