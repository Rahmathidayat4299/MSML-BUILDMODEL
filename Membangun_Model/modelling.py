# modelling.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# Set MLflow ke tracking lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Graduate Basic Local")

# Aktifkan autolog Scikit-Learn (CORE REQUIREMENT)
mlflow.sklearn.autolog()

# Load dataset hasil preprocessing dengan path relatif
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_path = os.path.join(current_dir, "Graduate_indicators_preprocessing")

train_df = pd.read_csv(os.path.join(preprocess_path, "train_processed.csv"))
test_df = pd.read_csv(os.path.join(preprocess_path, "test_processed.csv"))

X_train = train_df.drop(columns="Status")
y_train = train_df["Status"]
X_test = test_df.drop(columns="Status")
y_test = test_df["Status"]

# Jalankan training dengan autolog (TANPA log manual)
with mlflow.start_run(run_name="RandomForest_Baseline_Local"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Autolog akan menangkap metrik secara otomatis

print("Training selesai. Cek MLflow Tracking UI di http://127.0.0.1:5000")