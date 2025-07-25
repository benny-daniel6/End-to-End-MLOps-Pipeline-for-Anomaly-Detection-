import pandas as pd
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
import os
import numpy as np
import joblib

exp_name='transaction_anamoly_detection'
mlflow.set_experiment(exp_name)
model_dir="model"
os.makedirs(model_dir, exist_ok=True)
model_path=os.path.join(model_dir, "transaction_anomaly_model.joblib")
data_path="data/transactions.csv"


def train_model():

    print("Starting a new training run for transaction anamoly detection.....")
    with mlflow.start_run() as run:
        run_id=run.info.run_id
        print(f"MLFlow run ID : {run_id}")

        n_estimators=100
        contamination=0.01
        random_state=42
        input_example = pd.DataFrame([[500.0, 13.5]], columns=["Transaction_Amount", "TimeOfDay"])


        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("data_path", data_path)
        print("Logged parameters to MLFlow")

        print(F"Loading data from {data_path}...")


        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Data file {data_path} not found.")
            print("Please run 'dvc pull' or 'python prepare_data.py' first.")


        x= df[["Transaction_Amount", "TimeOfDay"]]
        print(f"Data generated with {len(df)} samples.")

        print("Training Isolation Forest model...")
        iso_forest=IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            
        )
        iso_forest.fit(x)
        print("Model training completed.")

        predictions=iso_forest.predict(x)
        anomalies_detected=np.sum(predictions==-1)
        print(f"Logged metric: Detected {anomalies_detected} anamolies.")

        mlflow.sklearn.log_model(iso_forest, name="transaction_anomaly_model", input_example=input_example,signature=mlflow.models.infer_signature(x, iso_forest.predict(x)))
        print("Model has been logged to MLFlow as an artifact")

        joblib.dump(iso_forest, model_path)
        print(f"Model saved to {model_path}")

    print ("Training and logging completed successfully.")
if __name__ == "__main__":
    train_model()
    print("Training script completed.")




    
