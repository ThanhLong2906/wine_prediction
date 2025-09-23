import pandas as pd
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data(path="data/prices.csv"):
    df = pd.read_csv(path)
    X = df[["size", "color", "freshness"]]  # giả định các feature
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train():
    X_train, X_test, y_train, y_test = load_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # log với MLflow
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "linear_regression")

    # lưu model ra thư mục model/
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print(f"Model trained, MSE={mse}")


if __name__ == "__main__":
    mlflow.set_experiment("rose_price")
    with mlflow.start_run():
        train()
