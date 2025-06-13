import dagshub
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Set up DagsHub for MLflow tracking
dagshub.init(
    repo_owner="Rahmathidayat4299", repo_name="MSML-BUILDMODEL", mlflow=True
)
mlflow.set_experiment("Diabetes Health Basics")

# Enable MLflow autologging for LightGBM
mlflow.lightgbm.autolog()

# Load data
train_df = pd.read_csv("Graduate_Health_Indicators_Preprocessing/train_processed.csv")
test_df = pd.read_csv("Graduate_Health_Indicators_Preprocessing/test_processed.csv")

X_train = train_df.drop(columns="Status")
y_train = train_df["Status"]
X_test = test_df.drop(columns="Status")
y_test = test_df["Status"]

# Train a basic LightGBM model
with mlflow.start_run(run_name="LightGBM_Baseline"):
    model = LGBMClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Log test accuracy
    preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, preds)
    mlflow.log_metric("test_accuracy", test_accuracy)