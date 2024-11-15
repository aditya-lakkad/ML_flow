"""
Description:
    this scripts fetches the registered model over mlflow ui.
Date :
    15th Nov,2024
Author:
    Aditya Lakkad
Usage:
    -
"""
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
logged_model = 'models:/Prediction_DT/Staging'  # if it is in Production then append production instead of staging

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = [[1.0, 0.0, 0.0, 0.0, 0.0, 4.98, 360.0, 1.0, 2.0, 8.60]]  # sample input data
print(loaded_model.predict(pd.DataFrame(data)))