# ======================================================
# Modelling (CI SAFE - MLflow Project)
# ======================================================

import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ======================================================
# Path Configuration
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "telco_churn_preprocessing")

TRAIN_PATH = os.path.join(DATA_DIR, "telco_train_processed.csv")
TEST_PATH  = os.path.join(DATA_DIR, "telco_test_processed.csv")

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ======================================================
# Load Dataset
# ======================================================
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]

X_test = test_df.drop("Churn", axis=1)
y_test = test_df["Churn"]

# ======================================================
# Training (NO start_run, NO set_experiment)
# ======================================================
model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ======================================================
# Metrics
# ======================================================
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
}

mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("solver", "liblinear")
mlflow.log_param("max_iter", 1000)
mlflow.log_metrics(metrics)

# ======================================================
# Log Model
# ======================================================
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

# ======================================================
# Artifacts
# ======================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.tight_layout()

cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
mlflow.log_artifact(cm_path)

report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(classification_report(y_test, y_pred, zero_division=0))
mlflow.log_artifact(report_path)

metric_json_path = os.path.join(ARTIFACT_DIR, "metric_summary.json")
with open(metric_json_path, "w") as f:
    json.dump(metrics, f, indent=4)
mlflow.log_artifact(metric_json_path)

# ======================================================
# Save run_id
# ======================================================
run_id = mlflow.active_run().info.run_id
run_id_path = os.path.join(BASE_DIR, "run_id.txt")

with open(run_id_path, "w") as f:
    f.write(run_id)

mlflow.log_artifact(run_id_path)

print("===================================")
print("Training Completed Successfully")
print("Run ID:", run_id)
print("Metrics:", metrics)
print("===================================")
