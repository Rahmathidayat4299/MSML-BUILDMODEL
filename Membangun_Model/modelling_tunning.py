import json
import time
import os # Import os module

import dagshub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --- KONFIGURASI MLFLOW & DAGSHUB ---
# Pastikan Anda menginisialisasi DagsHub hanya sekali di awal skrip
dagshub.init(
    repo_owner="Rahmathidayat4299", repo_name="MSML-BUILDMODEL", mlflow=True
)

# Set experiment untuk pelacakan di Dagshub
mlflow.set_experiment("Graduate  Indicators")

# --- LOAD DATA ---
train_df = pd.read_csv("Graduate_indicators_preprocessing/train_processed.csv")
test_df = pd.read_csv("Graduate_indicators_preprocessing/test_processed.csv")


X_train = train_df.drop(columns="Status")
y_train = train_df["Status"]
X_test = test_df.drop(columns="Status")
y_test = test_df["Status"]

# Prepare an input example for MLflow model signature
input_example = X_train.head(1)

# --- FUNGSI OBJECTIVE UNTUK OPTUNA ---
def objective(trial, run_id):
    params = {
        "boosting_type": "gbdt",
        "device": "cpu",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name=f"LightGBM_Trial_{run_id}"):
        model = LGBMClassifier(**params)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Measure prediction time
        start_predict = time.time()
        preds = model.predict(X_test)
        predict_time = time.time() - start_predict

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Log parameters and metrics (ini akan dilog ke Dagshub karena tracking URI utama)
        mlflow.log_params(params)
        mlflow.log_param("trial_id", run_id)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.log_metric("predict_time_seconds", predict_time)

        # Log metrics as JSON
        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "train_time_seconds": train_time,
            "predict_time_seconds": predict_time,
        }
        mlflow.log_text(json.dumps(metrics_dict, indent=2), "metric_info.json")

        # --- Bagian yang Diperbarui: Log model ke lokal ---
        # Simpan URI tracking yang aktif saat ini (Dagshub)
        original_tracking_uri = mlflow.get_tracking_uri()

        try:
            # Set URI tracking ke server MLflow lokal untuk logging model
            # Pastikan Anda menjalankan `mlflow ui` di terminal terpisah
            mlflow.set_tracking_uri("http://127.0.0.1:5001")
            
            # Log model ke server MLflow lokal
            mlflow.sklearn.log_model(
                sk_model=model,
                # Gunakan 'name' alih-alih 'artifact_path' untuk mengatasi peringatan
                name=f"model_trial_{run_id}",
                input_example=input_example,
                # Opsional: daftar ke MLflow Model Registry lokal
                registered_model_name="LGBM_Graduate_Indicators_Model" 
            )
            print(f"Model untuk Trial {run_id} berhasil dilog ke MLflow lokal (http://127.0.0.1:5001).")
        except Exception as e:
            print(f"Gagal melog model untuk Trial {run_id} ke MLflow lokal: {e}")
        finally:
            # Sangat penting: Kembalikan URI tracking ke Dagshub setelah logging model selesai
            mlflow.set_tracking_uri(original_tracking_uri)
        # --- Akhir Bagian Log Model Lokal ---

        # Log confusion matrix (akan dilog ke Dagshub)
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Log feature importance plot (akan dilog ke Dagshub)
        fig, ax = plt.subplots(figsize=(10, 6))
        importances = model.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importances)[::-1]
        ax.bar(range(len(importances)), importances[indices], align="center")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(feature_names[indices], rotation=90)
        ax.set_title(f"Feature Importance - Trial {run_id}")
        plt.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        return acc

# --- JALANKAN OPTUNA ---
# Anda mungkin ingin menyimpan study ini ke DB agar persisten jika menjalankan ulang
# study = optuna.create_study(direction="maximize", study_name="graduate_lgbm_tuning", storage="sqlite:///optuna.db")
study = optuna.create_study(direction="maximize", study_name="graduate_lgbm_tuning")

# Jalankan optimasi untuk 15 percobaan
print("Memulai optimasi Optuna...")
for i in range(15):
    study.optimize(lambda trial: objective(trial, trial.number + 1), n_trials=1)

print("\nOptimasi Selesai!")
print(f"Jumlah percobaan yang diselesaikan: {len(study.trials)}")
print(f"Trial terbaik: {study.best_trial.value}")
print(f"Hyperparameter terbaik: {study.best_trial.params}")