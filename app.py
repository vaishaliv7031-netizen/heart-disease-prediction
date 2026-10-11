import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction")
st.write("Logistic Regression using resting blood pressure (`trestbps`) and cholesterol (`chol`).")
st.warning("Educational demo only — this model is not a medical diagnostic tool.")

uploaded_file = st.file_uploader("Upload your dataset CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload your `dataset.csv` file to begin. The CSV must contain `trestbps`, `chol`, and `target` columns.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

required = ["trestbps", "chol", "target"]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()

df = df[required].copy()
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

if df.empty or df["target"].nunique() != 2:
    st.error("After removing missing/non-numeric rows, the dataset must contain both target classes (0 and 1).")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"Usable rows: {len(df)}")

X = df[["trestbps", "chol"]]
y = df["target"].astype(int)

if not set(y.unique()).issubset({0, 1}):
    st.error("This demo expects a binary target encoded as 0 and 1.")
    st.stop()

test_size = st.slider("Test-set percentage", min_value=10, max_value=40, value=20, step=5)
if y.value_counts().min() < 2:
    st.error("Each target class needs at least two rows for a stratified train/test split.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Test-set performance")
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 score": f1_score(y_test, y_pred, zero_division=0),
}
try:
    metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)
except ValueError:
    metrics["ROC-AUC"] = float("nan")

cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.3f}" if pd.notna(value) else "N/A")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"], ax=ax
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("ROC curve")
    if y_test.nunique() == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics['ROC-AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("ROC curve requires both classes in the test set.")

st.subheader("Try a sample prediction")
c1, c2 = st.columns(2)
with c1:
    blood_pressure = st.number_input("Resting blood pressure (trestbps)", min_value=50, max_value=250, value=120)
with c2:
    cholesterol = st.number_input("Cholesterol (chol)", min_value=80, max_value=700, value=200)

if st.button("Predict", type="primary"):
    sample = pd.DataFrame({"trestbps": [blood_pressure], "chol": [cholesterol]})
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0, 1])
    st.write(f"Model output: **{prediction}**")
    st.write(f"Estimated probability for target 1: **{probability:.1%}**")
    st.caption("This probability reflects this dataset/model only and should not guide medical decisions.")
