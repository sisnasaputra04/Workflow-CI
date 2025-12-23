import os
import json
import mlflow
import pandas as pd
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ======================================================
# Load Processed Dataset
# ======================================================
def load_processed_data():
    """
    Memuat dataset Telco Churn yang sudah diproses
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, "telco_churn_preprocessing")
    train_csv = os.path.join(data_dir, "telco_train_processed.csv")
    test_csv  = os.path.join(data_dir, "telco_test_processed.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(
            "Processed dataset tidak ditemukan.\n"
            f"- Expected train: {train_csv}\n"
            f"- Expected test : {test_csv}"
        )

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    target = "Churn"
    if target not in train_df.columns:
        raise ValueError(f"Kolom target '{target}' tidak ditemukan")

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    print(f"[INFO] Dataset Loaded | Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ======================================================
# Training with MLflow (CI SAFE)
# ======================================================
def run_training(X_train, y_train, X_test, y_test):
    """
    Training model RandomForest dengan MLflow Autolog
    """

    # Gunakan tracking URI dari environment (CI)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[INFO] MLFLOW_TRACKING_URI: {tracking_uri or '(default)'}")

    # Pastikan tidak ada conflict run CI
    os.environ.pop("MLFLOW_RUN_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

    # Aktifkan Autolog
    mlflow.autolog()

    # Set experiment (AMAN DI CI)
    mlflow.set_experiment("Experiment_CI_Telco_Churn")

    print("[INFO] Training RandomForestClassifier (CI)...")
    with mlflow.start_run(run_name="rf_telco_ci") as run:

        # Simpan run_id untuk workflow
        run_id_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_id.txt")
        with open(run_id_path, "w", encoding="utf-8") as f:
            f.write(run.info.run_id)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

        # Training
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        # Log metric manual
        mlflow.log_metric("test_accuracy_manual", float(acc))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1", float(f1))

        # WAJIB: simpan model di artifact_path="model"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Artifact tambahan
        outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        metrics_path = os.path.join(outputs_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "run_id": run.info.run_id
                },
                f,
                indent=2
            )

        mlflow.log_artifacts(outputs_dir, artifact_path="outputs")

        print("\n[SUCCESS] Training Selesai")
        print(f"[INFO] Run ID        : {run.info.run_id}")
        print(f"[INFO] Accuracy     : {acc:.4f}")
        print(f"[INFO] F1 Score     : {f1:.4f}")
        print("[INFO] Model tersimpan di artifacts/model/")
        print("[INFO] run_id.txt siap dibaca workflow CI")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_processed_data()
    run_training(X_train, y_train, X_test, y_test)
